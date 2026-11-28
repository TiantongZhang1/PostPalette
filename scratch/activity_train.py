"""From-scratch daily activities: CPU default, per-batch recovery, validation-only abstention."""
import argparse
import copy
import json
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from .config import DATA, ARTIFACTS
from .daily_activity import ActivityCNN, CLASSES, LABELS, SIZE, decide, adjust_probabilities
from .train import save_checkpoint, save_json


def metrics(probabilities, labels):
    matrix=np.zeros((len(CLASSES),len(CLASSES)),dtype=np.int64)
    np.add.at(matrix,(labels,probabilities.argmax(1)),1)
    tp=np.diag(matrix);f1=2*tp/np.maximum(matrix.sum(0)+matrix.sum(1),1)
    top3=np.argsort(-probabilities,axis=1)[:,:3]
    known=labels<len(LABELS)
    return {'accuracy':float(tp.sum()/max(matrix.sum(),1)), 'macro_f1':float(f1.mean()),
            'supported_8_accuracy':float((probabilities.argmax(1)[known]==labels[known]).mean()) if known.any() else None,
            'always_other_accuracy':float((~known).mean()),
            'supported_8_macro_f1':float(f1[:len(LABELS)].mean()),
            'top3_accuracy':float((top3==labels[:,None]).any(1).mean()),
            'per_class_f1':dict(zip(CLASSES,f1.tolist())), 'confusion_matrix':matrix.tolist()}


@torch.inference_mode()
def predict(model,images):
    model.eval();result=[];device=next(model.parameters()).device
    for offset in range(0,len(images),48):
        x=torch.from_numpy(images[offset:offset+48].copy()).to(device).float()/255
        result.append(model(x).softmax(1).cpu().numpy())
    return np.concatenate(result)


@torch.no_grad()
def refresh_batchnorm(model, training_images):
    """EMA parameters need normalization statistics measured with those same parameters."""
    layers=[layer for layer in model.modules() if isinstance(layer,torch.nn.BatchNorm2d)]
    if not layers:return
    model.train()
    for layer in model.modules():
        if isinstance(layer,torch.nn.Dropout):layer.eval()
    for layer in layers:layer.reset_running_stats();layer.momentum=None
    device=next(model.parameters()).device
    for offset in range(0,min(512,len(training_images)),32):
        x=torch.from_numpy(training_images[offset:offset+32].copy()).to(device).float()/255
        model(x)
    model.eval()


def calibrate(probabilities, labels, target=.8, minimum=12, margin=.15):
    thresholds=[];supports=[]
    order=np.argsort(-probabilities,axis=1);winner=order[:,0]
    margins=probabilities[np.arange(len(labels)),winner]-probabilities[np.arange(len(labels)),order[:,1]]
    for category in range(len(LABELS)):
        options=[]
        for threshold in np.arange(.50,.981,.02):
            selected=(winner==category)&(probabilities[:,category]>=threshold)&(margins>=margin)
            n=int(selected.sum());precision=float((labels[selected]==category).mean()) if n else 0
            if n>=minimum and precision>=target:options.append((n,float(threshold),precision))
        if options:
            n,threshold,precision=max(options,key=lambda r:(r[0],r[2]));thresholds.append(threshold)
            supports.append({'accepted':n,'precision':precision})
        else:thresholds.append(1.01);supports.append({'accepted':0,'precision':None})
    return {'thresholds':thresholds,'margin':margin,'target_validation_precision':target,
            'minimum_validation_suggestions':minimum,'validation_support':dict(zip(LABELS,supports)),
            'notice':'Finite validation-set selection rule; scores are not calibrated correctness probabilities or an OOD guarantee.'}


def selective_metrics(probabilities,labels,rule):
    results=[decide(p,rule['thresholds'],rule['margin']) for p in probabilities]
    accepted=np.array([r['accepted'] for r in results]);predicted=probabilities.argmax(1)
    known=labels<len(LABELS);other=~known
    return {'suggestions':int(accepted.sum()), 'coverage':float(accepted.mean()),
            'suggestion_precision':float((predicted[accepted]==labels[accepted]).mean()) if accepted.any() else None,
            'supported_activity_coverage':float(accepted[known].mean()) if known.any() else None,
            'other_false_suggestion_rate':float(accepted[other].mean()) if other.any() else None,
            'per_class':{name:{'true_images':int((labels==i).sum()),'suggestions':int(((predicted==i)&accepted).sum()),
                                'correct_suggestions':int(((labels==i)&(predicted==i)&accepted).sum())} for i,name in enumerate(LABELS)}}


def select_other_bias(probabilities, labels):
    candidates=[]
    for bias in (-1.,-.5,0.,.5,1.,1.5,2.,2.5):
        score=metrics(adjust_probabilities(probabilities,bias),labels)
        candidates.append((score['macro_f1'],-abs(bias),bias,score))
    _,_,bias,score=max(candidates,key=lambda row:row[:2])
    return bias,score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--minutes',type=float,default=20);p.add_argument('--steps',type=int,default=9000)
    p.add_argument('--batch-size',type=int,default=24);p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--normalization',choices=('group','batch'),default='batch')
    p.add_argument('--sampling',choices=('uniform','mixed_other'),default='mixed_other')
    p.add_argument('--run-name',choices=('daily_activity','daily_activity_bn'),default='daily_activity')
    p.add_argument('--resume',action='store_true');p.add_argument('--evaluate',action='store_true')
    p.add_argument('--validate',action='store_true');a=p.parse_args()
    if a.minutes<=0 or a.steps<1 or a.batch_size<1 or a.threads<1:p.error('Training limits and thread count must be positive')
    torch.set_num_threads(a.threads);torch.manual_seed(2026);torch.backends.cudnn.enabled=False;rng=random.Random(2026)
    folder=DATA/'daily_activity';manifest=json.loads((folder/'manifest.json').read_text())
    if manifest['labels']!=list(CLASSES) or manifest['size']!=SIZE:raise ValueError('Activity data configuration mismatch')
    data={s:np.load(folder/f'{s}.npz') for s in ('train','val')}
    images={s:d['images'] for s,d in data.items()};labels={s:d['labels'] for s,d in data.items()}
    directory=ARTIFACTS/a.run_name;directory.mkdir(parents=True,exist_ok=True)
    if not (a.resume or a.evaluate or a.validate) and ((directory/'last.pt').exists() or (directory/'best.pt').exists()):p.error('Use --resume to preserve existing training progress')
    r=None
    if a.resume or a.evaluate or a.validate:r=torch.load(directory/('best.pt' if a.evaluate or a.validate else 'last.pt'),map_location='cpu',weights_only=True)
    config=r.get('config',{'normalization':'group'}) if r else {'normalization':a.normalization}
    sampling=r.get('sampling','uniform' if config['normalization']=='group' else 'mixed_other') if r else a.sampling
    model=ActivityCNN(**config).to(a.device);averaged=copy.deepcopy(model).eval()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.001)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=3,min_lr=.00002)
    step=0;best=-1;history=[];elapsed=0.
    if a.resume or a.evaluate or a.validate:
        if r['manifest']!=manifest:raise ValueError('Data changed since checkpoint; do not resume on another split')
        model.load_state_dict(r['train_model']);averaged.load_state_dict(r['model']);optimizer.load_state_dict(r['optimizer'])
        scheduler.load_state_dict(r['scheduler']);step=r['step'];best=r['best_val'];history=r['history'];elapsed=r['training_seconds']
        rng.setstate(r['rng']);torch.set_rng_state(r['torch_rng'])
        if a.device=='cuda' and r.get('cuda_rng') is not None:torch.cuda.set_rng_state(r['cuda_rng'])
    if a.evaluate or a.validate:
        raw_validation=predict(averaged,images['val']);bias,validation_report=select_other_bias(raw_validation,labels['val'])
        validation=adjust_probabilities(raw_validation,bias);rule=calibrate(validation,labels['val']);rule['other_logit_bias']=bias
        validation_report.update({'step':step,'config':config,'sampling':sampling,'raw_macro_f1':best,
                                  'calibration':rule,'selective':selective_metrics(validation,labels['val'],rule)})
        save_json(directory/'validation_summary.json',validation_report)
        if a.validate:
            print(json.dumps({k:v for k,v in validation_report.items() if k not in ('confusion_matrix','calibration','per_class_f1')}),flush=True);return
        test=np.load(folder/'test.npz');raw_probabilities=predict(averaged,test['images'])
        probabilities=adjust_probabilities(raw_probabilities,bias)
        report=metrics(probabilities,test['labels'])
        report.update({'task':'daily_activity','step':step,'split':'test','images':len(test['labels']),
                       'validation_macro_f1':validation_report['macro_f1'],'raw_validation_macro_f1':best,'labels':CLASSES,'calibration':rule,
                       'config':config,'sampling':sampling,'raw_model_test':metrics(raw_probabilities,test['labels']),
                       'selective':selective_metrics(probabilities,test['labels'],rule),
                       'training_seconds':elapsed,'parquet_sha256':manifest['parquet_sha256']})
        save_json(directory/'calibration.json',{'step':step,**rule})
        save_json(directory/'test_metrics.json',report);print(json.dumps(report),flush=True);return
    members=[np.flatnonzero(labels['train']==i).tolist() for i in range(len(CLASSES))]
    if not all(members):raise ValueError('Missing training category')
    started=time.monotonic();print(f'Model origin: random weights; {sum(p.numel() for p in model.parameters())} parameters; device {a.device}; threads {a.threads}; resume/start step {step}',flush=True)
    try:
        while step<a.steps and time.monotonic()-started<a.minutes*60:
            if (directory/'STOP').exists():break
            # Give varied non-supported activities enough training exposure without letting them dominate.
            if sampling=='uniform':indices=[rng.choice(members[rng.randrange(len(CLASSES))]) for _ in range(a.batch_size)]
            else:indices=[rng.choice(members[len(LABELS) if rng.random()<.3 else rng.randrange(len(LABELS))]) for _ in range(a.batch_size)]
            x=torch.from_numpy(images['train'][indices].copy()).to(a.device).float()/255
            y=torch.from_numpy(labels['train'][indices]).to(a.device)
            theta=torch.zeros(len(x),2,3,device=a.device);theta[:,0,0]=.9+.2*torch.rand(len(x),device=a.device)
            theta[:,1,1]=theta[:,0,0];theta[:,0,0]*=torch.where(torch.rand(len(x),device=a.device)<.5,-1.,1.)
            theta[:,:,2]=.12*torch.rand(len(x),2,device=a.device)-.06
            x=F.grid_sample(x-.5,F.affine_grid(theta,x.shape,align_corners=False),align_corners=False)+.5
            mean=x.mean((2,3),keepdim=True)
            x=((x-mean)*(.8+.4*torch.rand(len(x),1,1,1,device=a.device))+mean)
            x=(x*(.85+.3*torch.rand(len(x),3,1,1,device=a.device))).clamp(0,1)
            model.train();optimizer.zero_grad(set_to_none=True)
            loss=F.cross_entropy(model(x),y,label_smoothing=.05)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss; stopping without retrying')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),3);optimizer.step();step+=1
            with torch.no_grad():
                for average,current in zip(averaged.parameters(),model.parameters()):average.lerp_(current,.01)
                for average,current in zip(averaged.buffers(),model.buffers()):
                    if average.is_floating_point():average.lerp_(current,.01)
                    else:average.copy_(current)
            improved=False
            if step%200==0 or step==1:
                refresh_batchnorm(averaged,images['train'])
                report=metrics(predict(averaged,images['val']),labels['val'])
                improved=report['macro_f1']>best;best=max(best,report['macro_f1']);scheduler.step(report['macro_f1'])
                entry={'step':step,'loss':float(loss.detach()),'val_accuracy':report['accuracy'],'val_macro_f1':report['macro_f1'],
                       'val_supported_8_macro_f1':report['supported_8_macro_f1'],'seconds_this_run':time.monotonic()-started,
                       'lr':optimizer.param_groups[0]['lr']}
                entry['bn_statistics_refreshed_from_training']=config['normalization']=='batch'
                history.append(entry);print(json.dumps(entry),flush=True)
            record={'task':'daily_activity','initialization':'random','step':step,'best_val':best,'size':SIZE,'labels':CLASSES,
                    'manifest':manifest,'config':config,'sampling':sampling,'model':averaged.state_dict(),'train_model':model.state_dict(),
                    'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'rng':rng.getstate(),
                    'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state() if a.device=='cuda' else None,
                    'history':history,'training_seconds':elapsed+time.monotonic()-started,'device':a.device,'batch_size':a.batch_size,'threads':a.threads}
            save_checkpoint(directory/'last.pt',record)
            if improved:save_checkpoint(directory/'best.pt',record)
    except KeyboardInterrupt:pass
    save_json(directory/'history.json',history)
    print(f'Stopped at {step}; best validation macro F1 {best}; seconds this run {time.monotonic()-started:.1f}',flush=True)


if __name__=='__main__':main()
