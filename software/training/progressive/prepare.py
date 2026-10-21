import train as t
import torch,json,copy,platform
from pathlib import Path
t.setup();t.a.seed();out=t.ROOT/'bs2';out.mkdir(exist_ok=True)
assert not (t.ROOT/'initial_fp32.pth').exists()
source=t.PRETRAINED;source_sha=t.TH
assert t.r.sha(source)==source_sha
c=torch.load(source,map_location='cpu',weights_only=False)
pretrained_epoch=c.get('epoch')
base=t.create_arraypolyp_variant('replk_tile',input_size=256)
base.load_state_dict(c['model'],strict=True);base.eval();base.switch_to_deploy()
m=t.model().cuda()
names=[n for n,q in m.named_modules() if isinstance(q,t.p.QConv)]
listed=[n for _,_,ns in t.GROUPS for n in ns]
assert len(names)==25 and len(listed)==len(set(listed))==25 and set(names)==set(listed)
with torch.no_grad():
 for n,q in m.named_modules():
  if isinstance(q,t.p.QConv):
   original=base.get_submodule(n.removeprefix('base.'))
   q.weight.copy_(original.weight.to(q.weight.device));q.bias.copy_(original.bias.to(q.bias.device))
c={'model':t.r.cpu(m.state_dict())}
for n,q in m.named_modules():
 if isinstance(q,t.p.QConv):
  assert torch.equal(q.weight.cpu(),c['model'][n+'.weight']) and torch.equal(q.bias.cpu(),c['model'][n+'.bias'])
  fresh=t.p.QConv(q,8).cuda()
  with torch.no_grad():q.log_scale.copy_(fresh.log_scale)
  q.bits=8
split=json.loads((t.SRC.parent/'data'/'kvasir_split.json').read_text());assert [len(split[k]) for k in ['train','validation','test']]==[720,80,200]
ds=t.r.dataset(split['train'],True);op=t.a.optimizer(m);t.transition(m,op,1,dict(next_batch=0),ds);t.observer(m,200)
state=t.r.cpu(m.state_dict());fp=t.r.fingerprint(state)
t.r.atomic_save(t.ROOT/'initial_fp32.pth',dict(model=state,pretrained_loaded=True,source_fp32_sha256=t.TH,seed=42,fingerprint=fp,weight_bits={n:8 for n in names},activation_bits=8))
files=[Path(t.__file__).resolve(),Path(__file__)]+list(t.SRC.glob('*.py'))+list((t.SRC/'tilepolyp').glob('*.py'))
cfg=dict(experiment='progressive_groupwise_W8_to_ternary_A8_noKD_FP32_bs2',epochs=200,batch_size=2,seed=42,samples_per_epoch=720,optimizer_steps=72000,initialization='exact25 FP32 folded Conv latent weights/biases; fresh W8 per-output scales for all25; never trained ternary best',source_initialization=str(source),source_initialization_sha256=source_sha,pretrained_checkpoint=str(t.PRETRAINED),pretrained_sha256=t.TH,pretrained_epoch=pretrained_epoch,initial_file_sha256=t.r.sha(t.ROOT/'initial_fp32.pth'),initial_fp32_qat_fingerprint=fp,source_hashes={str(f):t.r.sha(f) for f in files},split_sha256=t.r.sha(t.SRC.parent/'data'/'kvasir_split.json'),KD=False,kd_weight=0,loss='BCEWithLogits+softDice',groups=t.GROUPS,quantization='epochs1-10 all25 W8; groups convert to ternary at11/21/31/41/51/61/71/81;all53 existing activation boundaries always A8 signed[-127,127];FP32 latent,bias,accumulators,Add arithmetic;fakequant',observer='first64 augmented training recalibration at1 and each group change; EMA .99 epochs1-20 and 3epochs starting each change, otherwise frozen; eval never updates',optimizer='fresh AdamW foreachFalse weight_decay1e-5 including scales; clear only changed log_scale moments at each change',base_lr=2e-4,lr_schedule='original 5epoch warmup cosinefloor.02;each change3epochs min(original_lr,2e-5), then original cosine',gradient_clip=1.,augmentation='original PIL paired;SHA25642:epoch:index;order randperm seed42000+epoch',selection='final best only fullW2A8 epochs81-200 validation80 macro Dice threshold.5; phase best/last also retained;no earlystop',test_ood_access=False,amp=False,tf32=False,deterministic=True,threads=2,workers=0,versions=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,numpy=t.np.__version__),python=t.sys.executable)
t.write(t.ROOT,'shared_config.json',cfg);before=t.r.sha(t.ROOT/'initial_fp32.pth');rows=[]
# Separate process only: smoke every real switch; no formal optimizer/RNG is reused.
for ep in [1]+[start for start,_,_ in t.GROUPS]:
 prior={n:q.weight.detach().clone() for n,q in m.named_modules() if isinstance(q,t.p.QConv)};t.transition(m,op,ep,dict(next_batch=0),ds)
 assert all(torch.equal(prior[n],q.weight) for n,q in m.named_modules() if isinstance(q,t.p.QConv))
 t.observer(m,ep);m.train();x,y=t.batch(ds,[0,1],ep);op.zero_grad();loss=t.r.loss_fn(m(x),y);loss.backward();assert all(v.grad is not None and v.grad.isfinite().all() for v in m.parameters());torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);op.step();op.zero_grad(set_to_none=True)
 for n,q in m.named_modules():
  if isinstance(q,t.p.QConv):
   code=q.quant_weight()/q.log_scale.exp().clamp_min(1e-8);assert float(code.abs().max())<=(127 if q.bits==8 else 1)+1e-4 and torch.allclose(code,code.round(),atol=1e-4)
  if isinstance(q,t.p.Act8):
   q.adaptive.fill_(False);q.eval();code=q(torch.linspace(-200,200,401,device='cuda')*q.scale)/q.scale;assert int(q.abits)==8 and abs(float(code.min())+127)<1e-4 and abs(float(code.max())-127)<1e-4 and torch.allclose(code,code.round(),atol=1e-4)
 rows.append(dict(epoch=ep,phase=t.stage_name(ep),ternary_count=sum(q.bits==2 for q in m.modules() if isinstance(q,t.p.QConv)),loss=float(loss),finite_gradients=True,quant_ranges_pass=True,latent_unchanged_by_switch=True))
t.observer(m,200);mm=t.model().cuda();mm.load_state_dict(m.state_dict(),strict=True)
for n,q in mm.named_modules():
 if isinstance(q,t.p.QConv):q.bits=2
op2=t.a.optimizer(mm);op2.load_state_dict(copy.deepcopy(op.state_dict()));x,y=t.batch(ds,[2,3],81)
for obj,opt in [(m,op),(mm,op2)]:
 obj.train();opt.zero_grad();loss=t.r.loss_fn(obj(x),y);loss.backward();torch.nn.utils.clip_grad_norm_(obj.parameters(),1,error_if_nonfinite=True);opt.step()
assert t.r.fingerprint(m.state_dict())==t.r.fingerprint(mm.state_dict());assert t.r.sha(t.ROOT/'initial_fp32.pth')==before
t.write(t.ROOT,'smoke.json',dict(status='PASS',all25_FP32_masters_exact=True,group_coverage_exact25_no_duplicates=True,rows=rows,strict_resume_next_step_equal=True,formal_initial_unchanged=True,separate_process_from_formal=True,no_teacher_noKD=True));print('PROGRESSIVE_W2A8_SMOKE_PASS',flush=True)
