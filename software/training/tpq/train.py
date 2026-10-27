import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,json,time,copy,math,random,hashlib,argparse,signal,traceback,platform
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
SRC=Path(__file__).resolve().parents[2]/'core';ROOT=Path(os.environ['TERNSCOPE_RUN_DIR'])
sys.path.insert(0,str(SRC))
import training_utils as a
import topology as r
import quantization as p
from ternscope import create_ternscope
r.DATA=Path(os.environ['TERNSCOPE_DATA_ROOT'])
PRETRAINED=Path(os.environ['TERNSCOPE_FP32'])
TH=r.sha(PRETRAINED)
EPOCHS=200;LR=2e-4;KD=0.0;STOP=False

def stopping(*args):
 global STOP;STOP=True
signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGINT,stopping)

def act_forward(self,x):
 if self.observe:
  with torch.no_grad():self.peak.copy_(torch.maximum(self.peak,x.detach().abs().amax()))
  return x
 if self.training and bool(self.adaptive):
  with torch.no_grad():
   self.peak.mul_(.99).add_(x.detach().abs().amax(),alpha=.01)
   self.scale.copy_(self.peak.clamp_min(1e-5)/(2**(int(self.abits)-1)-1));self.updates.add_(1)
 if not self.enabled:return x
 s=self.scale.clone();limit=2**(int(self.abits)-1)-1;z=(x/s).clamp(-limit,limit)
 return (z+(z.round()-z).detach())*s
p.Act8.forward=act_forward

def model():
 m=p.QuantGraph(r.TernScopeDeploy().eval(),'int8')
 for q in m.modules():
  if isinstance(q,p.Act8):q.register_buffer('abits',torch.tensor(8));q.register_buffer('adaptive',torch.tensor(False));q.register_buffer('updates',torch.tensor(0,dtype=torch.int64))
 return m

def observer(m,ep):
 for q in m.modules():
  if isinstance(q,p.Act8):q.adaptive.fill_(ep<=20)
def buffers(m):return {n:(q.scale.detach().cpu().clone(),int(q.updates)) for n,q in m.named_modules() if isinstance(q,p.Act8)}
def lr(ep):return LR*(ep/5 if ep<=5 else .02+.98*.5*(1+math.cos(math.pi*(ep-5)/195)))
def write(out,n,v):r.write(out/n,v)
def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore(d):
 random.setstate(d['python']);np.random.set_state(d['numpy']);torch.set_rng_state(d['torch'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in d['cuda']])
def batch(ds,indices,ep):
 # Sample augmentation is independent of minibatch grouping and arm/GPU.
 state=random.getstate();items=[]
 for i in indices:
  token=hashlib.sha256(f'42:epoch:{ep}:train_index:{i}'.encode()).digest();random.seed(int.from_bytes(token[:8],'big'));items.append(ds[i])
 random.setstate(state)
 return torch.stack([v[0] for v in items]).cuda(),torch.stack([v[1] for v in items]).cuda()
def order(ep):return torch.randperm(720,generator=torch.Generator().manual_seed(42000+ep)).tolist()

def save(out,n,m,op,st):r.atomic_save(out/n,dict(model=m.state_dict(),optimizer=op.state_dict(),state=copy.deepcopy(st),rng=rng(),teacher_required=False,weight_bits={n:q.bits for n,q in m.named_modules() if isinstance(q,p.QConv)},activation_bits={n:int(q.abits) for n,q in m.named_modules() if isinstance(q,p.Act8)}))
def setup():
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True);a.seed()
def train(bs,gpu,resume):
 out=ROOT/f'bs{bs}';out.mkdir(exist_ok=True);cfg=json.loads((ROOT/'shared_config.json').read_text())
 assert os.environ['CUDA_VISIBLE_DEVICES']==gpu
 assert all(r.sha(Path(f))==h for f,h in cfg['source_hashes'].items()) and r.sha(ROOT/'initial_fp32.pth')==cfg['initial_file_sha256']
 m=model().cuda();op=a.optimizer(m)
 split=json.loads((SRC.parent/'data'/'kvasir_split.json').read_text());assert len(split['train'])==720 and len(split['validation'])==80
 ds=r.dataset(split['train'],True);val=r.dataset(split['validation'],False)
 if resume:
  c=torch.load(out/'last.pth',map_location='cpu',weights_only=False);m.load_state_dict(c['model']);op.load_state_dict(c['optimizer']);st=c['state'];restore(c['rng'])
  for n,q in m.named_modules():
   if isinstance(q,p.QConv):q.bits=c['weight_bits'][n]
 else:
  assert not (out/'last.pth').exists()
  c=torch.load(ROOT/'initial_fp32.pth',map_location='cpu',weights_only=False);m.load_state_dict(c['model']);assert r.fingerprint(m.state_dict())==cfg['initial_fp32_qat_fingerprint'];assert not op.state
  initial=r.evaluate(m,val,'cuda');write(out,'initial_validation.json',dict(student=initial))
  st=dict(epoch=1,next_batch=0,steps=0,examples=0,seg_sum=0.,kd_sum=0.,teacher_seg_sum=0.,epoch_examples=0,epoch_train_seconds=0.,history=[],best_dice=-1.,best_epoch=0,stage_bests={})
  save(out,'last.pth',m,op,st)
 write(out,'run_config.json',dict(**cfg,batch=bs,gpu_uuid=gpu,pid=os.getpid(),optimizer_steps_target=200*(720//bs),launch_time=time.time()))
 print(json.dumps(dict(event='STARTED',pid=os.getpid(),batch=bs,gpu=gpu,resume=resume,epoch=st['epoch'])),flush=True)
 jobstart=time.time()
 while st['epoch']<=200:
  ep=st['epoch'];transition(m,op,ep,st,ds);observer(m,ep);m.train();frozen=buffers(m) if not adapting(ep) else None
  for group in op.param_groups:group['lr']=effective_lr(ep)
  indices=order(ep);stepstart=time.time()
  for j in range(st['next_batch'],720//bs):
   tb=time.time();ii=indices[j*bs:(j+1)*bs];x,y=batch(ds,ii,ep);op.zero_grad(set_to_none=True)
   tls=torch.tensor(0.,device=x.device)
   z=m(x);seg=r.loss_fn(z,y);kl=torch.tensor(0.,device=x.device);loss=seg
   assert bool(torch.isfinite(loss));loss.backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
   assert all(v.grad is not None and bool(v.grad.isfinite().all()) for v in m.parameters())
   op.step();st['steps']+=1;st['examples']+=bs;st['next_batch']=j+1;st['seg_sum']+=float(seg.detach())*bs;st['kd_sum']+=float(kl.detach())*bs;st['teacher_seg_sum']+=float(tls)*bs;st['epoch_examples']+=bs;st['epoch_train_seconds']+=time.time()-tb
   if st['steps']%60==0 or STOP:
    save(out,'last.pth',m,op,st);write(out,'heartbeat.json',dict(pid=os.getpid(),epoch=ep,batch_index=j+1,optimizer_steps=st['steps'],examples=st['examples'],updated_unix=time.time(),elapsed_process_seconds=time.time()-jobstart))
   if STOP:write(out,'paused.json',dict(reason='signal',epoch=ep,next_batch=j+1));return
  assert st['epoch_examples']==720
  tv=time.time();score=r.evaluate(m,val,'cuda');val_seconds=time.time()-tv
  assert KD==0
  acts=buffers(m)
  if frozen:assert all(torch.equal(v[0],acts[n][0]) and v[1]==acts[n][1] for n,v in frozen.items())
  assert all(bool(v.isfinite().all()) for v in m.parameters())
  row=dict(epoch=ep,phase=stage_name(ep),weight_bits={n:q.bits for n,q in m.named_modules() if isinstance(q,p.QConv)},activation_bits=8,batch=bs,optimizer_steps=st['steps'],examples=st['examples'],train_seg=st['seg_sum']/720,train_kd=st['kd_sum']/720,train_total=(st['seg_sum']+KD*st['kd_sum'])/720,teacher_train_seg=st['teacher_seg_sum']/720,validation=score,lr=effective_lr(ep),train_seconds=st['epoch_train_seconds'],validation_seconds=val_seconds,seconds=st['epoch_train_seconds']+val_seconds,kd_enabled=False,all_student_gradients_finite=True,last_gradient_norm_before_clip=float(gn),epoch_order_sha256=hashlib.sha256(json.dumps(indices).encode()).hexdigest(),augmentation_seed_scheme='shared SHA256(seed42,epoch,index)',observer_frozen=not adapting(ep),observer_scales={n:float(v[0]) for n,v in acts.items()},observer_updates={n:v[1] for n,v in acts.items()})
  st['history'].append(row)
  stage=stage_name(ep)
  prior=st['stage_bests'].get(stage,dict(dice=-1))
  if score['dice']>prior['dice']:
   st['stage_bests'][stage]=dict(epoch=ep,dice=score['dice'],validation=score);save(out,stage+'_best.pth',m,op,st)
  save(out,stage+'_last.pth',m,op,st)
  if ep>=81 and score['dice']>st['best_dice']:st['best_dice']=score['dice'];st['best_epoch']=ep;save(out,'best.pth',m,op,st)
  st.update(epoch=ep+1,next_batch=0,seg_sum=0.,kd_sum=0.,teacher_seg_sum=0.,epoch_examples=0,epoch_train_seconds=0.)
  save(out,'last.pth',m,op,st);write(out,'history.json',st['history']);write(out,'progress.json',dict(pid=os.getpid(),latest=row,best_epoch=st['best_epoch'],best_dice=st['best_dice'],remaining_seconds_estimate=(200-ep)*sum(v['seconds'] for v in st['history'][-5:])/len(st['history'][-5:]),updated_unix=time.time()))
  print(json.dumps(dict(event='epoch',batch=bs,epoch=ep,dice=score['dice'],iou=score['iou'],seg=row['train_seg'],kd=row['train_kd'],seconds=row['seconds'],best_dice=st['best_dice'])),flush=True)
 assert st['steps']==200*(720//bs) and st['examples']==144000
 write(out,'results.json',dict(stage_bests=st['stage_bests'],status='COMPLETED',epochs=200,batch=bs,steps=st['steps'],examples=st['examples'],best_epoch=st['best_epoch'],best_val_dice=st['best_dice'],last_validation=score,total_epoch_gpu_wall_seconds=sum(v['seconds'] for v in st['history']),kd_enabled=False,pretrained_checkpoint_unchanged=r.sha(PRETRAINED)==TH,test_ood_access=False,last_sha256=r.sha(out/'last.pth'),best_sha256=r.sha(out/'best.pth')))
 print('COMPLETED',bs,flush=True)

GROUPS=[
 (11,'stage4_down5',['base.down5.deploy_conv','base.stage4.0.spatial.deploy_conv','base.stage4.0.channel_mixer.conv']),
 (21,'stage3_down4',['base.down4.deploy_conv']+[f'base.stage3.{i}.{part}' for i in range(2) for part in ['spatial.deploy_conv','channel_mixer.conv']]),
 (31,'stage2_down3',['base.down3.deploy_conv']+[f'base.stage2.{i}.{part}' for i in range(2) for part in ['spatial.deploy_conv','channel_mixer.conv']]),
 (41,'stage1_down2',['base.down2.deploy_conv','base.stage1.0.spatial.deploy_conv','base.stage1.0.channel_mixer.conv']),
 (51,'skip_encoders',[f'base.skip{i}_encode.0' for i in range(1,5)]),
 (61,'high_semantic',['base.high_refine.spatial.deploy_conv','base.semantic_encode.0']),
 (71,'detail',['base.detail_refine.spatial.deploy_conv']),
 (81,'first_head',['base.down1.0','base.segmentation_head'])]
def stage_name(ep):
 k=sum(ep>=start for start,_,_ in GROUPS)
 return 'full_W2A8' if k==8 else f'conversion_{k}_of_8'
def adapting(ep):return ep<=20 or any(start<=ep<=start+2 for start,_,_ in GROUPS)
def observer(m,ep):
 for q in m.modules():
  if isinstance(q,p.Act8):q.adaptive.fill_(adapting(ep))
def effective_lr(ep):return min(lr(ep),2e-5) if any(start<=ep<=start+2 for start,_,_ in GROUPS) else lr(ep)
def transition(m,op,ep,st,ds):
 if ep not in [1]+[start for start,_,_ in GROUPS] or st['next_batch']!=0:return
 state=rng();m.eval();target={n for start,_,names in GROUPS if ep>=start for n in names};changed=[]
 for n,q in m.named_modules():
  if isinstance(q,p.QConv):
   bits=2 if n in target else 8
   if q.bits!=bits:
    fresh=p.QConv(q,bits).to(q.weight.device)
    with torch.no_grad():q.log_scale.copy_(fresh.log_scale)
    op.state.pop(q.log_scale,None);q.bits=bits;changed.append(n)
 for q in m.modules():
  if isinstance(q,p.Act8):q.abits.fill_(8);q.peak.zero_();q.observe=True
 with torch.no_grad():
  for i in range(0,64,2):x,_=batch(ds,[i,i+1],ep);m(x)
 for q in m.modules():
  if isinstance(q,p.Act8):q.scale.copy_(q.peak.clamp_min(1e-5)/127);q.observe=False
 restore(state)
 write(ROOT,f'bs2/transition_{ep}.json',dict(epoch=ep,phase=stage_name(ep),changed_layers=changed,train_calibration_indices=list(range(64)),rng_restored=True,observer_EMA_epochs=20 if ep==1 else 3,adaptation_lr=None if ep==1 else 2e-5,weight_bits={n:q.bits for n,q in m.named_modules() if isinstance(q,p.QConv)},activation_bits={n:int(q.abits) for n,q in m.named_modules() if isinstance(q,p.Act8)}))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--batch',type=int,choices=[2]);ap.add_argument('--gpu');ap.add_argument('--resume',action='store_true');args=ap.parse_args()
 setup()
 try:
  train(args.batch,args.gpu,args.resume)
 except Exception:
  out=ROOT/f'bs{args.batch}' if args.batch else ROOT;out.mkdir(exist_ok=True);write(out,'failure.json',dict(traceback=traceback.format_exc(),unix=time.time()));raise
