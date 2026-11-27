"""Train scratch expression/action classifiers, with per-batch checkpoints and a time cap."""

import argparse,copy,json,random,time
import numpy as np
import torch
from torch.nn import functional as F
from .config import ARTIFACTS,DATA
from .people_networks import ImageClassifier
from .train import save_checkpoint,save_json


@torch.inference_mode()
def evaluate(model,images,labels,classes):
    model.eval();matrix=np.zeros((classes,classes),dtype=np.int64);top3_correct=0
    for offset in range(0,len(images),64):
        batch=torch.from_numpy(images[offset:offset+64].copy()).to(next(model.parameters()).device).float()/255
        logits=model(batch)
        prediction=logits.argmax(1).cpu().numpy()
        top3=logits.topk(min(3,classes),dim=1).indices.cpu().numpy()
        top3_correct+=int((top3==labels[offset:offset+64,None]).any(1).sum())
        np.add.at(matrix,(labels[offset:offset+64],prediction),1)
    tp=np.diag(matrix);f1=2*tp/np.maximum(matrix.sum(0)+matrix.sum(1),1)
    return {'accuracy':float(tp.sum()/max(matrix.sum(),1)),'top3_accuracy':top3_correct/max(len(labels),1),'macro_f1':float(f1.mean()),'per_class_f1':f1.tolist(),'confusion_matrix':matrix.tolist()}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('task',choices=('expression','action'))
    p.add_argument('--minutes',type=float,default=15);p.add_argument('--steps',type=int,default=20000)
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--resume',action='store_true');p.add_argument('--evaluate',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(2026);torch.backends.cudnn.enabled=False;rng=random.Random(2026)
    folder=DATA/('actions' if a.task=='action' else 'expression');manifest=json.loads((folder/'manifest.json').read_text())
    arrays={split:np.load(folder/f'{split}.npz') for split in ('train','val','test')}
    images={s:r['images'] for s,r in arrays.items()};labels={s:r['labels'] for s,r in arrays.items()}
    directory=ARTIFACTS/a.task;directory.mkdir(exist_ok=True)
    if not (a.resume or a.evaluate) and ((directory/'last.pt').exists() or (directory/'best.pt').exists()):
        p.error('Existing checkpoint found; use --resume instead of overwriting training progress')
    config={'classes':len(manifest['labels']),'channels':images['train'].shape[1]}
    model=ImageClassifier(**config).to(a.device);optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    averaged=copy.deepcopy(model).eval();step=0;best=-1;history=[]
    if a.resume or a.evaluate:
        r=torch.load(directory/('best.pt' if a.evaluate else 'last.pt'),map_location='cpu',weights_only=True)
        assert r['manifest']==manifest
        model.load_state_dict(r['train_model']);averaged.load_state_dict(r['model']);optimizer.load_state_dict(r['optimizer'])
        step,best,history=r['step'],r['best_val'],r['history'];rng.setstate(r['rng']);torch.set_rng_state(r['torch_rng'])
        if a.device=='cuda' and r.get('cuda_rng') is not None:torch.cuda.set_rng_state(r['cuda_rng'])
    if a.evaluate:
        report=evaluate(averaged,images['test'],labels['test'],config['classes'])
        report.update({'split':'test','step':step,'classes':manifest['labels']})
        save_json(directory/'test_metrics.json',report);print(json.dumps(report),flush=True);return
    counts=np.bincount(labels['train'],minlength=config['classes']);weights=torch.tensor((counts.max()/np.maximum(counts,1))**.5,device=a.device,dtype=torch.float32).clamp(max=4)
    started=time.monotonic();print(f'{a.task} scratch training on {a.device}, {sum(x.numel() for x in model.parameters())} parameters; starting {step}',flush=True)
    try:
        while step<a.steps and time.monotonic()-started<a.minutes*60:
            if (directory/'STOP').exists() or (ARTIFACTS/'people_pipeline/STOP').exists():break
            indices=[rng.randrange(len(images['train'])) for _ in range(a.batch_size)]
            x=torch.from_numpy(images['train'][indices].copy()).to(a.device).float()/255
            y=torch.from_numpy(labels['train'][indices]).to(a.device)
            flip=torch.rand(len(x),device=a.device)<.5;x[flip]=x[flip].flip(-1)
            # Mild photometric variation preserves object/action/expression labels.
            x=(x*(.85+.3*torch.rand(len(x),1,1,1,device=a.device))).clamp(0,1)
            model.train();optimizer.zero_grad(set_to_none=True)
            loss=F.cross_entropy(model(x),y,weight=weights,label_smoothing=.05);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2);optimizer.step();step+=1
            with torch.no_grad():
                for average,current in zip(averaged.parameters(),model.parameters()):average.lerp_(current,.01)
            improved=False
            if step%200==0 or step==1:
                report=evaluate(averaged,images['val'],labels['val'],config['classes'])
                improved=report['macro_f1']>best;best=max(best,report['macro_f1'])
                entry={'step':step,'loss':float(loss.detach()),'val_accuracy':report['accuracy'],'val_macro_f1':report['macro_f1']}
                history.append(entry);print(json.dumps(entry),flush=True)
            r={'task':a.task,'initialization':'random','step':step,'best_val':best,'config':config,'manifest':manifest,
               'labels':manifest['labels'],'model':averaged.state_dict(),'train_model':model.state_dict(),'optimizer':optimizer.state_dict(),
               'rng':rng.getstate(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state() if a.device=='cuda' else None,'history':history}
            save_checkpoint(directory/'last.pt',r)
            if improved:save_checkpoint(directory/'best.pt',r)
    except KeyboardInterrupt:pass
    save_json(directory/'history.json',history);print(f'{a.task} stopped at {step}; best validation macro F1 {best}',flush=True)


if __name__=='__main__':main()
