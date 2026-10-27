"""TernScope-Net deployment topology and shared TPQ training utilities."""
import os, json, random, hashlib, math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ternscope.training import SegmentationSample, PolypSegmentationDataset, SegmentationMetrics
DATA = Path(os.environ.get("TERNSCOPE_DATA_ROOT", "data/Kvasir-SEG"))

def seed(v=42):random.seed(v);np.random.seed(v);torch.manual_seed(v)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,obj):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));os.replace(tmp,p)


def cpu(x):
 if torch.is_tensor(x):return x.detach().cpu().clone()
 if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
 if isinstance(x,list):return [cpu(v) for v in x]
 if isinstance(x,tuple):return tuple(cpu(v) for v in x)
 return x


def fingerprint(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()


def atomic_save(p,obj):
 p=Path(p);tmp=p.with_suffix('.tmp');torch.save(cpu(obj),tmp);os.replace(tmp,p)


def conv(ci,co,k,s=1):return nn.Conv2d(ci,co,k,stride=s,padding=k//2 if k%2 else 0,bias=True)


def cba(ci,co,k,s=1,act=True):return nn.Sequential(conv(ci,co,k,s),nn.Identity(),nn.ReLU6() if act else nn.Identity())


class Spatial(nn.Module):
 def __init__(self,c,k):super().__init__();self.deploy_conv=conv(c,c,k)
 def forward(self,x):return self.deploy_conv(x)


class Mixer(nn.Module):
 def __init__(self,c):super().__init__();self.conv=conv(c,c,1)
 def forward(self,x):return self.conv(x)


class Block(nn.Module):
 def __init__(self,c,part,k):
  super().__init__();self.part=part;self.spatial=Spatial(part,k);self.channel_mixer=Mixer(c)
 def forward(self,x):
  z=F.relu6(self.spatial(x[:,:self.part]));z=torch.cat((z,x[:,self.part:]),1) if x.shape[1]>self.part else z
  return F.relu6(self.channel_mixer(z))


class Down(nn.Module):
 def __init__(self,ci,co):super().__init__();self.deploy_conv=conv(ci,co,2,2)
 def forward(self,x):return F.relu6(self.deploy_conv(x))


class Refine(nn.Module):
 def __init__(self):super().__init__();self.spatial=Spatial(16,3)
 def forward(self,x):return F.relu6(self.spatial(x))


class TernScopeDeploy(nn.Module):
 """Branch-fused TernScope-Net with stable checkpoint tensor names."""
 def __init__(self):
  super().__init__()
  self.down1=cba(3,16,3,2);self.down2=Down(16,32);self.down3=Down(32,64);self.down4=Down(64,96);self.down5=Down(96,128)
  self.stage1=nn.Sequential(Block(32,16,3));self.stage2=nn.Sequential(Block(64,16,3),Block(64,16,5));self.stage3=nn.Sequential(Block(96,16,3),Block(96,16,7));self.stage4=nn.Sequential(Block(128,32,9))
  self.skip1_encode=cba(32,16,1,act=False);self.skip2_encode=cba(64,16,1,act=False);self.skip3_encode=cba(96,16,1,act=False);self.skip4_encode=cba(128,16,1,act=False)
  self.high_refine=Refine();self.semantic_encode=cba(16,16,1,act=False);self.detail_refine=Refine();self.segmentation_head=conv(16,1,1)
  self.add_q=nn.ModuleList([nn.Identity() for _ in range(3)])
  for name,m in self.named_modules():
   if isinstance(m,nn.Conv2d):
    nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='relu');nn.init.zeros_(m.bias)
  nn.init.xavier_normal_(self.segmentation_head.weight,gain=1.0)
 def forward(self,x):
  x1=self.stage1(self.down2(self.down1(x)));s1=self.skip1_encode(x1)
  x2=self.stage2(self.down3(x1));s2=self.skip2_encode(x2);x3=self.stage3(self.down4(x2));x4=self.stage4(self.down5(x3));s3=self.skip3_encode(x3);s4=self.skip4_encode(x4)
  rz=lambda a,b:F.interpolate(a,size=b.shape[-2:],mode='nearest')
  z=self.add_q[1](self.add_q[0](s2+rz(s3,s2))+rz(s4,s2));z=self.high_refine(z)
  z=self.detail_refine(self.add_q[2](s1+rz(self.semantic_encode(z),s1)))
  return F.interpolate(self.segmentation_head(z),size=x.shape[-2:],mode='nearest')


def dataset(ids,augment):
 d=DATA
 return PolypSegmentationDataset([SegmentationSample(dataset='Kvasir',image=d/'images'/i,mask=d/'masks'/i,sample_id=i) for i in ids],256,augment)


def batch(ds,indices,dev):
 items=[ds[i] for i in indices];return torch.stack([x[0] for x in items]).to(dev),torch.stack([x[1] for x in items]).to(dev)


def loss_fn(z,y):
 p=z.sigmoid();dims=(1,2,3)
 return F.binary_cross_entropy_with_logits(z,y)+(1-(2*(p*y).sum(dims)+1)/(p.sum(dims)+y.sum(dims)+1)).mean()


@torch.no_grad()
def evaluate(m,ds,dev):
 m.eval();met=SegmentationMetrics(.5);low=math.inf;high=-math.inf;empty=0
 for i in range(0,len(ds),2):
  x,y=batch(ds,list(range(i,min(len(ds),i+2))),dev);z=m(x);met.update(z.cpu(),y.cpu());low=min(low,float(z.min()));high=max(high,float(z.max()));empty+=int((z.flatten(1).amax(1)<0).sum())
 d=met.compute();return {**{k:d[k] for k in ('count','dice','iou','boundary_f1')},'logit_min':low,'logit_max':high,'empty_predictions':empty}

