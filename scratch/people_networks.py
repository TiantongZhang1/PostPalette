"""Randomly initialized residual CNNs; no external weights or learned processors."""

import torch
from torch import nn
from torch.nn import functional as F


class Residual(nn.Module):
    def __init__(self, source, target, stride=1):
        super().__init__()
        self.body=nn.Sequential(nn.Conv2d(source,target,3,stride,padding=1,bias=False),nn.GroupNorm(8,target),nn.SiLU(),
                                nn.Conv2d(target,target,3,padding=1,bias=False),nn.GroupNorm(8,target))
        self.skip=nn.Conv2d(source,target,1,stride,bias=False) if source!=target or stride!=1 else nn.Identity()
    def forward(self,x):return F.silu(self.body(x)+self.skip(x))


class Backbone(nn.Module):
    def __init__(self,channels=3):
        super().__init__()
        self.stages=nn.ModuleList([Residual(channels,24,2),Residual(24,48,2),Residual(48,96,2),Residual(96,128,2)])
    def forward(self,x):
        outputs=[]
        for stage in self.stages:x=stage(x);outputs.append(x)
        return outputs


class ImageClassifier(nn.Module):
    def __init__(self,classes,channels=3):
        super().__init__();self.backbone=Backbone(channels)
        self.head=nn.Sequential(nn.Linear(256,128),nn.SiLU(),nn.Dropout(.25),nn.Linear(128,classes))
    def forward(self,x):
        x=self.backbone(x)[-1]
        return self.head(torch.cat((x.mean((2,3)),x.amax((2,3))),1))


class PersonPoseNet(nn.Module):
    def __init__(self):
        super().__init__();self.backbone=Backbone()
        self.lateral=nn.Conv2d(96,64,1);self.top=nn.Conv2d(128,64,1)
        self.refine=Residual(64,64)
        self.heat=nn.Sequential(nn.Conv2d(64,64,3,padding=1),nn.SiLU(),nn.Conv2d(64,1,1))
        self.geometry=nn.Sequential(nn.Conv2d(64,64,3,padding=1),nn.SiLU(),nn.Conv2d(64,55,1))
        nn.init.constant_(self.heat[-1].bias,-2.19)
    def forward(self,x):
        stages=self.backbone(x)
        feature=self.refine(self.lateral(stages[2])+F.interpolate(self.top(stages[3]),size=stages[2].shape[-2:],mode='nearest'))
        return self.heat(feature),self.geometry(feature)


def overlap(a,b):
    x1,y1=max(a[0],b[0]),max(a[1],b[1]);x2,y2=min(a[2],b[2]),min(a[3],b[3])
    intersection=max(x2-x1,0)*max(y2-y1,0)
    return intersection/max((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection,1e-6)


@torch.inference_mode()
def detections(outputs,size=192,threshold=.3):
    heat,geometry=outputs
    probabilities=heat.sigmoid()
    peaks=probabilities*probabilities.eq(F.max_pool2d(probabilities,3,1,1))
    result=[]
    for batch in range(len(heat)):
        values,indices=peaks[batch,0].flatten().topk(min(40,peaks[batch,0].numel()))
        people=[];grid=heat.shape[-1];stride=size/grid
        for confidence,index in zip(values.cpu().tolist(),indices.cpu().tolist()):
            if confidence<threshold:continue
            y,x=divmod(index,grid);g=geometry[batch,:,y,x].cpu()
            width,height=g[:2].clamp(-3,4).exp().tolist()
            ox,oy=g[2:4].sigmoid().tolist();cx,cy=(x+ox)*stride,(y+oy)*stride
            width*=stride;height*=stride
            box=[max(0,cx-width/2),max(0,cy-height/2),min(size,cx+width/2),min(size,cy+height/2)]
            if min(box[2]-box[0],box[3]-box[1])<5 or any(overlap(box,p['box'])>.5 for p in people):continue
            keypoints=g[4:38].sigmoid().reshape(17,2)
            visible=g[38:55].sigmoid().tolist()
            points=[[box[0]+float(k[0])*(box[2]-box[0]),box[1]+float(k[1])*(box[3]-box[1]),visible[i]] for i,k in enumerate(keypoints)]
            people.append({'box':box,'score':confidence,'keypoints':points})
        result.append(people)
    return result
