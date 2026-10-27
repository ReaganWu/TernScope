"""Read frozen artifacts only; emit independent integer oracle and RTL memory images."""
from pathlib import Path
import json,hashlib,argparse
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rne(v,s):
 q=np.floor_divide(v,1<<s);r=v-q*(1<<s)
 return q+((r>(1<<(s-1)))|((r==(1<<(s-1)))&(q&1))).astype(np.int64) if s else v

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--artifacts',type=Path,required=True);ap.add_argument('--out',default='data');a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 e=a.artifacts;g=json.loads((e/'graph.json').read_text());p=np.load(e/'parameters.npz');nodes=g['nodes'];by={n['name']:n for n in nodes}
 def hx(name,b,width=16):
  b=bytes(b);assert len(b)%width==0
  (out/name).write_text(''.join(b[i:i+width][::-1].hex()+'\n' for i in range(0,len(b),width)))
 weights=b'';M=[];B=[];meta=[]
 for n in nodes:
  w=p[n['key']+'_W2'];wb=p[n['key']+'_Wbyte'];dec=np.stack([(w>>s)&3 for s in (0,2,4,6)],axis=-1).reshape(wb.shape).astype(np.int8)
  assert not np.any(dec==3);assert np.array_equal(np.where(dec==2,-1,dec),wb)
  meta.append(dict(**n,param_base=len(M),weight_byte_base=len(weights),weight_bytes=w.nbytes));weights+=w.tobytes();M.extend(p[n['key']+'_M'].tolist());B.extend(p[n['key']+'_B'].tolist())
 assert len(weights)==62080 and len(M)==1104
 hx('weights.hex128',weights);(out/'weights.bin').write_bytes(weights)
 (out/'params_M.hex32').write_text(''.join(f'{int(x)&0xffffffff:08x}\n' for x in M));(out/'params_B.hex48').write_text(''.join(f'{int(x)&((1<<48)-1):012x}\n' for x in B))
 lut=p['edge_LUTs'];hx('lut.hex8',lut.tobytes(),1)
 samples=[]
 for sample in range(4):
  gold=np.load(e/f'golden_validation_{sample}.npz');env={'input':gold['input']};hx(f'inputs_frame{sample}.hex128',gold['input'].transpose(1,2,0).tobytes());checks=[]
  def expand(s):
   z=env[s['node']][s['offset']*16:(s['offset']+s['count'])*16]
   return np.repeat(np.repeat(z,1<<s['ry'],1),1<<s['rx'],2)
  for n in nodes:
   if n['kind']=='add':
    z=[expand(s).astype(np.int64) for s in n['sources']];acc=np.clip(rne(z[0]*n['add_coefficients'][0]+z[1]*n['add_coefficients'][1],n['add_shift']),-127,127).astype(np.int32);v=acc.astype(np.int8)
   else:
    s=n['sources'][0];hh=env[s['node']].shape[1]*(1<<s['ry']);ww=env[s['node']].shape[2]*(1<<s['rx']);code=np.zeros((n['ci'],hh,ww),np.int8)
    for s in n['sources']:
     z=lut[s['lut']][expand(s).astype(np.uint8)];lo=s['start']*16;count=min(z.shape[0],n['ci']-lo);code[lo:lo+count]=z[:count]
    kh,kw=n['kernel'];sy,sx=n['stride'];ph,pw=n['padding'];pad=np.pad(code,((0,0),(ph,ph),(pw,pw)));win=np.lib.stride_tricks.sliding_window_view(pad,(kh,kw),axis=(1,2))[:,::sy,::sx]
    panel=win.transpose(1,2,3,4,0).reshape(-1,n['K']).astype(np.int32);w=p[n['key']+'_W'].transpose(2,3,1,0).reshape(n['K'],n['shape'][0]).astype(np.int32);acc=(panel@w).reshape(n['shape'][1],n['shape'][2],n['shape'][0]).transpose(2,0,1)
    assert np.array_equal(code,gold[n['key']+'_input'])
    v=np.clip(rne(acc.astype(np.int64)*p[n['key']+'_M'][:n['shape'][0],None,None]+p[n['key']+'_B'][:n['shape'][0],None,None],n['shift']),-127,127).astype(np.int8)
   assert np.array_equal(acc,gold[n['key']+'_acc']),(sample,n['name'],'acc');assert np.array_equal(v,gold[n['key']+'_out']),(sample,n['name'],'out');env[n['name']]=v
   packed=np.pad(v.transpose(1,2,0),((0,0),(0,0),(0,n['storage_c']-v.shape[0]))).tobytes();name=f'expected_frame{sample}_{n["key"]}.hex128';hx(name,packed);checks.append(dict(node=n['id'],file=name,bytes=len(packed),sha256=hashlib.sha256(packed).hexdigest(),acc_exact=True,output_exact=True))
  final=np.repeat(np.repeat(env[g['final']['source']],g['final']['repeat'],1),g['final']['repeat'],2);assert np.array_equal(final,gold['final']);hx(f'expected_frame{sample}_final.hex128',final.transpose(1,2,0).tobytes())
  samples.append(dict(sample=sample,golden_sha256=sha(e/f'golden_validation_{sample}.npz'),nodes=checks))
 report=dict(checkpoint=g['checkpoint_sha256'],graph_sha256=sha(e/'graph.json'),parameters_sha256=sha(e/'parameters.npz'),weight_sha256=hashlib.sha256(weights).hexdigest(),nodes=meta,frames=samples,eight_frame_order=[0,1,2,3,0,1,2,3],layout='HWC packed storage_c; each pixel channel-block16 forms little-endian128b row. RGB is dense HWC3, no padding. W2[kb,nb,inputlane,packed4outputbytes].',status='passed',checked_nodes=28,checked_samples=4)
 (out/'metadata.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:report[k] for k in ('status','checked_nodes','checked_samples','eight_frame_order')},indent=2))
if __name__=='__main__':main()
