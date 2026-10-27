"""Ternary weight and A8 activation operators for the TernScope deployment graph."""
import copy
import torch
from torch import nn
from torch.nn import functional as F

class Act8(nn.Module):
 def __init__(self):
  super().__init__();self.register_buffer('scale',torch.tensor(1.0/127));self.observe=False;self.enabled=True;self.register_buffer('peak',torch.tensor(0.0))
 def forward(self,x):
  if self.observe:
   self.peak.copy_(torch.maximum(self.peak,x.detach().abs().amax()))
   return x
  if not self.enabled:return x
  v=(x/self.scale).clamp(-127,127)
  return (v+(v.round()-v).detach())*self.scale
 def freeze(self):self.scale.copy_(self.peak.clamp_min(1e-6)/127);self.observe=False


class QConv(nn.Module):
 def __init__(self,c,bits):
  super().__init__();self.weight=nn.Parameter(c.weight.detach().clone());self.bias=nn.Parameter(c.bias.detach().clone()) if c.bias is not None else None
  self.stride,self.padding,self.dilation,self.groups=c.stride,c.padding,c.dilation,c.groups
  self.bits=bits;self.quant_weights=True;self.input_q=Act8();self.output_q=Act8()
  w=self.weight.detach();s=w.abs().amax((1,2,3),keepdim=True)/127 if bits==8 else w.abs().mean((1,2,3),keepdim=True)*1.5
  if bits==2:
   for _ in range(8):
    codes=(w/s.clamp_min(1e-8)).round().clamp(-1,1)
    s=(w*codes).sum((1,2,3),keepdim=True)/codes.square().sum((1,2,3),keepdim=True).clamp_min(1)
  self.log_scale=nn.Parameter(s.clamp_min(1e-8).log())
 def quant_weight(self):
  s=self.log_scale.exp().clamp_min(1e-8);limit=127 if self.bits==8 else 1
  v=(self.weight/s).clamp(-limit,limit)
  # STE through round; clipped straight-through weight gradient, LSQ-style scale derivative.
  return (v+(v.round()-v).detach())*s
 def forward(self,x):
  w=self.quant_weight() if self.quant_weights else self.weight
  return self.output_q(F.conv2d(self.input_q(x),w,self.bias,self.stride,self.padding,self.dilation,self.groups))


class QuantGraph(nn.Module):
 def __init__(self,base,mode):
  super().__init__();self.base=copy.deepcopy(base);self.mode=mode;self.add_q=nn.ModuleList([Act8() for _ in range(3)])
  self.coverage=[]
  self.base.aux_coarse_head=nn.Identity();self.base.aux_boundary_head=nn.Identity()
  for name,c in list(self.base.named_modules()):
   if isinstance(c,nn.Conv2d):
    bits=8 if mode=='int8' or name in ('down1.0','segmentation_head') else 2
    parent,_,leaf=name.rpartition('.');obj=self.base.get_submodule(parent) if parent else self.base
    setattr(obj,leaf,QConv(c,bits));self.coverage.append({'name':name,'weight_bits_storage':bits,'levels':255 if bits==8 else 3,'weights':c.weight.numel(),'scales':c.out_channels,'bias':c.out_channels,'kernel':list(c.kernel_size)})
  assert len(self.coverage)==25
 def forward(self,x):
  b=self.base;x1=b.stage1(b.down2(b.down1(x)));s1=b.skip1_encode(x1)
  x2=b.stage2(b.down3(x1));s2=b.skip2_encode(x2)
  x3=b.stage3(b.down4(x2));x4=b.stage4(b.down5(x3));s3=b.skip3_encode(x3);s4=b.skip4_encode(x4)
  rz=lambda x,y:F.interpolate(x,size=y.shape[-2:],mode='nearest')
  hi=self.add_q[1](self.add_q[0](s2+rz(s3,s2))+rz(s4,s2))
  hi=b.high_refine(hi);sem=rz(b.semantic_encode(hi),s1)
  detail=b.detail_refine(self.add_q[2](s1+sem))
  return F.interpolate(b.segmentation_head(detail),size=x.shape[-2:],mode='nearest')
 def observe(self):
  for m in self.modules():
   if isinstance(m,Act8):m.observe=True;m.peak.zero_()
   if isinstance(m,QConv):m.quant_weights=False
 def freeze(self):
  for m in self.modules():
   if isinstance(m,Act8):m.freeze()
   if isinstance(m,QConv):m.quant_weights=True
 def disable(self):
  for m in self.modules():
   if isinstance(m,Act8):m.enabled=False
   if isinstance(m,QConv):m.quant_weights=False
 def enable(self):
  for m in self.modules():
   if isinstance(m,Act8):m.enabled=True
   if isinstance(m,QConv):m.quant_weights=True

