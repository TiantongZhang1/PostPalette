"""Scratch center-based person detection and per-person keypoints, resumable each batch."""

import argparse,copy,json,math,random,time
import numpy as np
import torch
from torch.nn import functional as F
from .config import DATA,ARTIFACTS
from .people_networks import PersonPoseNet,detections,overlap
from .train import save_checkpoint,save_json


def convert(row,size,flip=False):
    scale=size/max(row['width'],row['height'])
    px=(size-round(row['width']*scale))//2;py=(size-round(row['height']*scale))//2
    people=[]
    for person in row['people']:
        x,y,w,h=person['bbox'];box=[x*scale+px,y*scale+py,(x+w)*scale+px,(y+h)*scale+py]
        points=np.array(person['keypoints'],dtype=np.float32).reshape(17,3)
        points[:,0]=points[:,0]*scale+px;points[:,1]=points[:,1]*scale+py
        if flip:
            box=[size-box[2],box[1],size-box[0],box[3]]
            points[:,0]=size-1-points[:,0]
            points=points[[0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15]]
        people.append({'box':box,'points':points})
    crowds=[]
    for x,y,w,h in row['crowds']:
        box=[x*scale+px,y*scale+py,(x+w)*scale+px,(y+h)*scale+py]
        if flip:box=[size-box[2],box[1],size-box[0],box[3]]
        crowds.append(box)
    return people,crowds


def targets(rows,size,flips):
    grid=size//8
    heat=np.zeros((len(rows),1,grid,grid),np.float32);valid=np.ones_like(heat)
    geometry=np.zeros((len(rows),55,grid,grid),np.float32)
    mask=np.zeros((len(rows),1,grid,grid),np.float32)
    jointmask=np.zeros((len(rows),34,grid,grid),np.float32)
    for i,(row,flip) in enumerate(zip(rows,flips)):
        people,crowds=convert(row,size,flip)
        for box in crowds:
            x1,y1,x2,y2=np.clip(np.array(box)/8,0,grid).astype(int)
            valid[i,0,y1:y2,x1:x2]=0
        # Favor larger annotations when centers quantize to the same feature cell.
        for person in sorted(people,key=lambda p:(p['box'][2]-p['box'][0])*(p['box'][3]-p['box'][1])):
            x1,y1,x2,y2=person['box'];w=max(x2-x1,1);h=max(y2-y1,1)
            cx,cy=(x1+x2)/16,(y1+y2)/16
            x,y=min(int(cx),grid-1),min(int(cy),grid-1)
            radius=int(np.clip(min(w,h)/24,1,4));sigma=max(radius/2,.7)
            for dy in range(-radius,radius+1):
                for dx in range(-radius,radius+1):
                    if 0<=x+dx<grid and 0<=y+dy<grid:
                        heat[i,0,y+dy,x+dx]=max(heat[i,0,y+dy,x+dx],math.exp(-(dx*dx+dy*dy)/(2*sigma*sigma)))
            heat[i,0,y,x]=1;valid[i,0,y,x]=1;mask[i,0,y,x]=1
            geometry[i,:4,y,x]=[math.log(w/8),math.log(h/8),cx-x,cy-y]
            points=person['points'];normalized=np.column_stack(((points[:,0]-x1)/w,(points[:,1]-y1)/h)).clip(0,1)
            geometry[i,4:38,y,x]=normalized.flatten()
            geometry[i,38:,y,x]=(points[:,2]==2)
            jointmask[i,:,y,x]=np.repeat(points[:,2]>0,2)
    return [torch.from_numpy(a) for a in (heat,valid,geometry,mask,jointmask)]


def objective(outputs,target):
    logits,g=outputs;heat,valid,truth,mask,joints=target
    p=logits.sigmoid().clamp(1e-5,1-1e-5);positive=heat.eq(1).float();negative=1-positive
    loss_heat=-(positive*(1-p).square()*p.log()+negative*(1-heat).pow(4)*p.square()*(1-p).log())*valid
    n=mask.sum().clamp_min(1)
    wh=(F.smooth_l1_loss(g[:,:2],truth[:,:2],reduction='none')*mask).sum()/n
    offset=(F.smooth_l1_loss(g[:,2:4].sigmoid(),truth[:,2:4],reduction='none')*mask).sum()/n
    pose=(F.smooth_l1_loss(g[:,4:38].sigmoid(),truth[:,4:38],reduction='none')*joints).sum()/joints.sum().clamp_min(1)
    visibility=(F.binary_cross_entropy_with_logits(g[:,38:],truth[:,38:],reduction='none')*mask).sum()/(17*n)
    return loss_heat.sum()/n+.2*wh+offset+2*pose+.2*visibility


def crowd_overlap(box,crowd):
    intersect=max(0,min(box[2],crowd[2])-max(box[0],crowd[0]))*max(0,min(box[3],crowd[3])-max(box[1],crowd[1]))
    return intersect/max((box[2]-box[0])*(box[3]-box[1]),1)


@torch.inference_mode()
def assess(model,images,rows,size,thresholds):
    model.eval();outputs=[]
    for offset in range(0,len(rows),16):
        x=torch.from_numpy(np.array(images[offset:offset+16],copy=True)).to(next(model.parameters()).device).float()/255
        outputs.extend(detections(model(x),size,.1))
    reports=[]
    for threshold in thresholds:
        tp=fp=fn=correct=0;errors=[];joint_correct=joint_total=0;positive_errors=[]
        groups={str(i):[] for i in range(4)}|{'4+':[]}
        for predictions,row in zip(outputs,rows):
            truth,crowds=convert(row,size);selected=[p for p in predictions if p['score']>=threshold]
            matches=set();counted=[]
            for person in selected:
                options=[(overlap(person['box'],target['box']),i) for i,target in enumerate(truth) if i not in matches]
                iou,index=max(options,default=(0,-1))
                if iou>=.5:
                    matches.add(index);tp+=1;counted.append(person)
                    points=truth[index]['points'];valid=points[:,2]>0
                    distance=np.linalg.norm(np.array(person['keypoints'])[:,:2]-points[:,:2],axis=1)
                    norm=max(truth[index]['box'][2]-truth[index]['box'][0],truth[index]['box'][3]-truth[index]['box'][1])
                    joint_correct+=int(((distance<.2*norm)&valid).sum());joint_total+=int(valid.sum())
                elif not any(crowd_overlap(person['box'],c)>.5 for c in crowds):fp+=1;counted.append(person)
            fn+=len(truth)-len(matches)
            if not crowds:
                error=abs(len(counted)-len(truth));errors.append(error);correct+=error==0
                if truth:positive_errors.append(error)
                groups[str(len(truth)) if len(truth)<4 else '4+'].append(error)
        reports.append({'threshold':threshold,'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'f1':2*tp/max(2*tp+fp+fn,1),
                        'count_mae':float(np.mean(errors)) if errors else None,'count_exact':correct/max(len(errors),1),
                        'positive_count_mae':float(np.mean(positive_errors)) if positive_errors else None,
                        'matched_person_pck_02':joint_correct/max(joint_total,1),'tp':tp,'fp':fp,'fn':fn,
                        'count_by_true_group':{k:{'images':len(v),'mae':float(np.mean(v)) if v else None,'exact':float(np.mean(np.array(v)==0)) if v else None} for k,v in groups.items()}})
    return max(reports,key=lambda r:r['f1'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--minutes',type=float,default=60)
    p.add_argument('--steps',type=int,default=30000);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu');p.add_argument('--resume',action='store_true');p.add_argument('--evaluate',action='store_true')
    a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(2026);torch.backends.cudnn.enabled=False;rng=random.Random(2026)
    folder=DATA/'people';manifest=json.loads((folder/'manifest.json').read_text());size=manifest['size']
    arrays={s:np.load(folder/f'{s}-{size}.npy',mmap_mode='r') for s in ('train','val','test')}
    rows={s:json.loads((folder/f'{s}.json').read_text()) for s in arrays}
    directory=ARTIFACTS/'person_pose';directory.mkdir(exist_ok=True)
    if not (a.resume or a.evaluate) and ((directory/'last.pt').exists() or (directory/'best.pt').exists()):
        p.error('Existing checkpoint found; use --resume instead of overwriting training progress')
    model=PersonPoseNet().to(a.device);averaged=copy.deepcopy(model).eval()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    step=0;best=-1;threshold=.3;history=[]
    if a.resume or a.evaluate:
        r=torch.load(directory/('best.pt' if a.evaluate else 'last.pt'),map_location='cpu',weights_only=True)
        assert r['manifest']==manifest
        model.load_state_dict(r['train_model']);averaged.load_state_dict(r['model']);optimizer.load_state_dict(r['optimizer'])
        step,best,threshold,history=r['step'],r['best_val'],r['threshold'],r['history'];rng.setstate(r['rng']);torch.set_rng_state(r['torch_rng'])
        if a.device=='cuda' and r.get('cuda_rng') is not None:torch.cuda.set_rng_state(r['cuda_rng'])
    if a.evaluate:
        report=assess(averaged,arrays['test'],rows['test'],size,[threshold]);report.update({'split':'test','step':step,'images':len(rows['test'])})
        save_json(directory/'test_metrics.json',report);print(json.dumps(report),flush=True);return
    started=time.monotonic();print(f'Person/pose scratch {a.device} training from step {step}; {sum(x.numel() for x in model.parameters())} parameters',flush=True)
    try:
        while step<a.steps and time.monotonic()-started<a.minutes*60:
            if (directory/'STOP').exists() or (ARTIFACTS/'people_pipeline/STOP').exists():break
            indices=[rng.randrange(len(rows['train'])) for _ in range(a.batch_size)];flip=[rng.random()<.5 for _ in indices]
            x=np.array(arrays['train'][indices],copy=True)
            for i,do_flip in enumerate(flip):
                if do_flip:x[i]=x[i,:,:,::-1]
            target=[t.to(a.device) for t in targets([rows['train'][i] for i in indices],size,flip)]
            x=torch.from_numpy(x).to(a.device).float()/255
            model.train();optimizer.zero_grad(set_to_none=True);loss=objective(model(x),target)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite detector loss; stopping')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();step+=1
            with torch.no_grad():
                for avg,param in zip(averaged.parameters(),model.parameters()):avg.lerp_(param,.01)
            improved=False
            if step%500==0 or step==1:
                report=assess(averaged,arrays['val'],rows['val'],size,[.15,.2,.25,.3,.4,.5,.6])
                threshold=report['threshold'];improved=report['f1']>best;best=max(best,report['f1'])
                entry={'step':step,'loss':float(loss.detach()),'val':report};history.append(entry);print(json.dumps(entry),flush=True)
            r={'task':'person_pose','initialization':'random','step':step,'size':size,'best_val':best,'threshold':threshold,'manifest':manifest,
               'model':averaged.state_dict(),'train_model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':rng.getstate(),
               'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state() if a.device=='cuda' else None,'history':history}
            save_checkpoint(directory/'last.pt',r)
            if improved:save_checkpoint(directory/'best.pt',r)
    except KeyboardInterrupt:pass
    save_json(directory/'history.json',history);print(f'Person/pose stopped at step {step}; best validation F1 {best}',flush=True)


if __name__=='__main__':main()
