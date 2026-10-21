"""W2A8 checkpoint -> fixed W2A8 graph, edge LUTs and independent integer oracle.

No training or checkpoint edits. The target performs view/window/convolution
in RTL; NumPy windows and materialized views here are oracle-only machinery.
Conv: sat127(RNE((acc*M+B)/2**24)), with output-domain signed bias.
ReLU6 is retained between producer output quantization and consumer input
quantization through exact finite-domain conversion tables.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rne(x, shift):
    import numpy as np
    x=np.asarray(x,dtype=np.int64)
    assert not np.any(x == np.iinfo(np.int64).min)
    if shift==0:return x
    mag=np.abs(x);q=mag>>shift;rem=mag&((1<<shift)-1);half=1<<(shift-1)
    q=q+((rem>half)|((rem==half)&((q&1)!=0)))
    return np.where(x<0,-q,q)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',type=Path,default=Path(__file__).resolve().parents[2]/'weights/ternscope_w2a8.pth')
    ap.add_argument('--trusted-checkpoint',action='store_true')
    ap.add_argument('--synthetic',action='store_true',help='Use deterministic synthetic inputs; no dataset or accuracy claims.')
    ap.add_argument('--data-root',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--samples',type=int,choices=[4],default=4)
    a=ap.parse_args()
    import numpy as np
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'software'))
    from runtime import load_w2a8, topology as r, quantization as p
    cp=a.checkpoint;before=sha(cp)
    output=a.output;output.mkdir(parents=True,exist_ok=False)
    model=load_w2a8(cp,trusted_checkpoint=a.trusted_checkpoint)
    codes={name:dict(codes=torch.round(q.weight.detach()/q.log_scale.detach().exp().clamp_min(1e-8)).clamp(-1,1).to(torch.int8),scale=q.log_scale.detach().exp().clamp_min(1e-8),bias=q.bias.detach()) for name,q in model.named_modules() if isinstance(q,p.QConv)}
    nodes=[];by={};arrays={};luts=[np.arange(256,dtype=np.uint8).view(np.int8)]
    lut_keys={luts[0].tobytes():0}
    xscale=float(model.get_submodule('base.down1.0').input_q.scale)
    by['input']=dict(shape=[3,256,256],scale=xscale,storage_c=3)
    def source(name,channels=None,offset=0,start=0,rx=0,ry=0,relu=False):
        c,h,w=by[name]['shape']
        count=(c-offset*16) if channels is None else channels
        assert name=='input' or count%16==0
        return dict(node=name,offset=offset,start=start,count=math.ceil(count/16),rx=rx,ry=ry,relu6=relu,lut=0)
    def lut(source,scale):
        if source['node']=='input':return 0
        src=by[source['node']]['scale']
        # Float32 is used only to enumerate the same finite conversion boundary
        # as CPU QAT. Hardware/runtime consume these frozen integer tables.
        q=torch.arange(-128,128,dtype=torch.float32)
        real=q*src
        if source['relu6']:real=real.clamp(0,6)
        z=(real/scale).clamp(-127,127).round().to(torch.int8).numpy()
        table=np.zeros(256,np.int8)
        table[np.arange(-128,128)&255]=z
        key=table.tobytes()
        if key not in lut_keys:lut_keys[key]=len(luts);luts.append(table)
        return lut_keys[key]
    def conv(short,sources):
        name='base.'+short;q=model.get_submodule(name);e=codes[name]
        co,ci,kh,kw=map(int,q.weight.shape)
        h=by[sources[0]['node']]['shape'][1]*(1<<sources[0]['ry'])
        w=by[sources[0]['node']]['shape'][2]*(1<<sources[0]['rx'])
        si=float(q.input_q.scale);so=float(q.output_q.scale)
        for s in sources:s['lut']=lut(s,si)
        oh=(h+2*q.padding[0]-kh)//q.stride[0]+1
        ow=(w+2*q.padding[1]-kw)//q.stride[1]+1
        weight=e['codes'].numpy().astype(np.int8)
        assert set(np.unique(weight)) <= {-1,0,1}
        assert np.array_equal(weight,torch.round(q.weight/q.log_scale.exp()).clamp(-1,1).to(torch.int8).numpy())
        k=ci*kh*kw;kp=(k+15)//16*16;cop=(co+15)//16*16
        m=np.rint(si*e['scale'].reshape(-1).double().numpy()/so*(1<<24)).astype(np.int64)
        b=np.rint(e['bias'].double().numpy()/so*(1<<24)).astype(np.int64)
        assert np.all((m>=0)&(m<2**31)) and np.all(np.abs(b)<2**47)
        assert np.all(k*127*m+np.abs(b)<2**63)
        matrix=np.zeros((kp,cop),np.int8)
        matrix[:k,:co]=weight.transpose(2,3,1,0).reshape(k,co)
        wb=matrix.reshape(kp//16,16,cop//16,16).transpose(0,2,1,3).copy()
        encode=np.where(wb==-1,2,wb).astype(np.uint8)
        packed=(encode[...,0::4]|(encode[...,1::4]<<2)|(encode[...,2::4]<<4)|(encode[...,3::4]<<6)).copy()
        key='n'+str(len(nodes)+1)
        arrays[key+'_Wbyte']=wb;arrays[key+'_W2']=packed
        arrays[key+'_W']=weight;arrays[key+'_M']=np.pad(m,(0,cop-co));arrays[key+'_B']=np.pad(b,(0,cop-co))
        n=dict(id=len(nodes)+1,key=key,name=name,kind='conv',shape=[co,oh,ow],storage_c=cop,
            ci=ci,kernel=[kh,kw],stride=list(q.stride),padding=list(q.padding),K=k,kb=kp//16,nb=cop//16,
            input_scale=si,scale=so,sources=sources,shift=24,
            output_clamp=[-127,127],weight_byte_rows=kp*cop//16,weight_W2_rows=kp*cop//64,
            max_M=int(m.max()),max_abs_output_bias_Q24=int(np.abs(b).max()),
            max_abs_acc_bound=k*127)
        nodes.append(n);by[name]=n;return name
    def block(x,index,c,part,k):
        spatial=conv(f'{index}.spatial.deploy_conv',[source(x,part,relu=True)])
        mix=conv(f'{index}.channel_mixer.conv',[source(spatial,part,relu=True),
            source(x,c-part,offset=part//16,start=part//16,relu=True)])
        return mix
    def add(i,sources):
        name='add_q.'+str(i);scale=float(model.get_submodule(name).scale)
        ratios=np.array([by[s['node']]['scale']/scale for s in sources])
        shift=min(24,math.floor(math.log2((2**31-1)/ratios.max())))
        coeff=np.rint(ratios*(1<<shift)).astype(np.int64)
        assert shift>=0 and np.all((coeff>=0)&(coeff<2**31))
        h=by[sources[0]['node']]['shape'][1]*(1<<sources[0]['ry'])
        w=by[sources[0]['node']]['shape'][2]*(1<<sources[0]['rx'])
        key='n'+str(len(nodes)+1)
        wb=np.eye(16,dtype=np.int8)[None,None,:,:]
        enc=wb.astype(np.uint8)
        packed=enc[...,0::4]|(enc[...,1::4]<<2)|(enc[...,2::4]<<4)|(enc[...,3::4]<<6)
        arrays[key+'_Wbyte']=wb;arrays[key+'_W2']=packed;arrays[key+'_W']=np.eye(16,dtype=np.int8)[:,:,None,None]
        arrays[key+'_M']=np.ones(16,np.int64);arrays[key+'_B']=np.zeros(16,np.int64)
        n=dict(id=len(nodes)+1,key=key,name=name,kind='add',shape=[16,h,w],storage_c=16,ci=16,
            kernel=[1,1],stride=[1,1],padding=[0,0],K=16,kb=1,nb=1,scale=scale,input_scale=scale,
            sources=sources,shift=0,add_shift=shift,add_coefficients=coeff.tolist(),output_clamp=[-127,127],
            weight_byte_rows=16,weight_W2_rows=4)
        nodes.append(n);by[name]=n;return name
    stem=conv('down1.0',[source('input')])
    d2=conv('down2.deploy_conv',[source(stem,relu=True)])
    x1=block(d2,'stage1.0',32,16,3)
    s1=conv('skip1_encode.0',[source(x1,relu=True)])
    d3=conv('down3.deploy_conv',[source(x1,relu=True)])
    x2=block(d3,'stage2.0',64,16,3);x2=block(x2,'stage2.1',64,16,5)
    s2=conv('skip2_encode.0',[source(x2,relu=True)])
    d4=conv('down4.deploy_conv',[source(x2,relu=True)])
    x3=block(d4,'stage3.0',96,16,3);x3=block(x3,'stage3.1',96,16,7)
    d5=conv('down5.deploy_conv',[source(x3,relu=True)])
    x4=block(d5,'stage4.0',128,32,9)
    s3=conv('skip3_encode.0',[source(x3,relu=True)])
    s4=conv('skip4_encode.0',[source(x4,relu=True)])
    a0=add(0,[source(s2),source(s3,rx=1,ry=1)])
    a1=add(1,[source(a0),source(s4,rx=2,ry=2)])
    hi=conv('high_refine.spatial.deploy_conv',[source(a1)])
    sem=conv('semantic_encode.0',[source(hi,relu=True)])
    a2=add(2,[source(s1),source(sem,rx=1,ry=1)])
    detail=conv('detail_refine.spatial.deploy_conv',[source(a2)])
    head=conv('segmentation_head',[source(detail,relu=True)])
    assert sum(n['kind']=='conv' for n in nodes)==25 and len(nodes)==28
    arrays['edge_LUTs']=np.stack(luts)
    graph=dict(schema='TERNSCOPE_FIXED_W2A8_GRAPH_V1',checkpoint_sha256=before,input_shape=[3,256,256],
        input_scale=xscale,nodes=nodes,final=dict(source=head,repeat=4,shape=[1,256,256]),
        lut_tables=len(luts),lut_sha256=hashlib.sha256(arrays['edge_LUTs'].tobytes()).hexdigest(),
        quantization='A8 [-127,127]; W{-1,0,+1}; Q24 output-domain bias; nearest-even; producer output quantization before edge ReLU6 and consumer input quantization',
        original_A8_scales_unchanged=True,training=False,all_network_RTL_verified=False)
    (output/'graph.json').write_text(json.dumps(graph,indent=2)+'\n')
    np.savez(output/'parameters.npz',**arrays)
    captured={};hooks=[]
    for name,q in model.named_modules():
        if isinstance(q,p.Act8):
            def hook(mod,inp,out,n=name):captured[n]=torch.round(out/mod.scale).to(torch.int8).squeeze(0).numpy().copy()
            hooks.append(q.register_forward_hook(hook))
    split=json.loads((Path(__file__).resolve().parents[2]/'software/data/kvasir_split.json').read_text())
    if a.synthetic:
        generator=torch.Generator().manual_seed(42)
        ds=[(torch.rand((3,256,256),generator=generator)*2-1,torch.zeros((1,256,256))) for _ in range(4)]
        split['validation']=[f'synthetic_{i}' for i in range(4)]
    else:
        if a.data_root is None:ap.error('--data-root is required unless --synthetic is used')
        r.DATA=a.data_root;ds=r.dataset(split['validation'],False)
    count=min(a.samples,len(ds));assert count>0;results=[];integer_metrics=r.SegmentationMetrics(.5);qat_metrics=r.SegmentationMetrics(.5)
    def expand(s,env):
        z=env[s['node']]
        z=z[s['offset']*16:(s['offset']+s['count'])*16]
        return np.repeat(np.repeat(z,1<<s['ry'],axis=1),1<<s['rx'],axis=2)
    def metrics(x,y):
        delta=x.astype(np.int64)-y.astype(np.int64)
        return dict(elements=int(delta.size),mismatches=int(np.count_nonzero(delta)),max_abs_code=int(np.abs(delta).max()))
    for sample in range(count):
        started=time.monotonic();x,y=r.batch(ds,[sample],'cpu')
        with torch.inference_mode():fp=model(x)
        qat_metrics.update(fp,y)
        initial=torch.round((x/model.get_submodule('base.down1.0').input_q.scale).clamp(-127,127)).to(torch.int8).squeeze(0).numpy()
        env={'input':initial};gold={'input':initial};checks=[]
        for n in nodes:
            if n['kind']=='add':
                z=[expand(s,env).astype(np.int64) for s in n['sources']]
                acc=np.clip(rne(z[0]*n['add_coefficients'][0]+z[1]*n['add_coefficients'][1],n['add_shift']),-127,127).astype(np.int32)
                out=acc.astype(np.int8);a_code=out
            else:
                c,h,w=by[n['sources'][0]['node']]['shape'];h<<=n['sources'][0]['ry'];w<<=n['sources'][0]['rx']
                a_code=np.zeros((n['ci'],h,w),np.int8)
                for s in n['sources']:
                    z=expand(s,env)
                    z=luts[s['lut']][z.astype(np.uint8)]
                    a_code[s['start']*16:min(n['ci'],(s['start']+s['count'])*16)]=z[:min(z.shape[0],n['ci']-s['start']*16)]
                kh,kw=n['kernel'];ph,pw=n['padding'];sy,sx=n['stride']
                padded=np.pad(a_code,((0,0),(ph,ph),(pw,pw)))
                windows=np.lib.stride_tricks.sliding_window_view(padded,(kh,kw),axis=(1,2))[:,::sy,::sx]
                panel=windows.transpose(1,2,3,4,0).reshape(-1,n['K']).astype(np.int32)
                weight=arrays[n['key']+'_W'];weights=weight.transpose(2,3,1,0).reshape(n['K'],n['shape'][0]).astype(np.int32)
                raw=panel@weights
                acc=raw.reshape(n['shape'][1],n['shape'][2],n['shape'][0]).transpose(2,0,1)
                # Scalar spatial dot-product checks are independent of panel addressing.
                for co,oy,ox in [(0,0,0),(n['shape'][0]-1,n['shape'][1]-1,n['shape'][2]-1)]:
                    val=0
                    for ci in range(n['ci']):
                        for ky in range(kh):
                            for kx in range(kw):
                                iy=oy*sy+ky-ph;ix=ox*sx+kx-pw
                                if 0<=iy<h and 0<=ix<w:val+=int(a_code[ci,iy,ix])*int(weight[co,ci,ky,kx])
                    assert val==int(acc[co,oy,ox]),(n['name'],co,oy,ox)
                m=arrays[n['key']+'_M'][:n['shape'][0],None,None]
                b=arrays[n['key']+'_B'][:n['shape'][0],None,None]
                out=np.clip(rne(acc.astype(np.int64)*m+b,n['shift']),-127,127).astype(np.int8)
                checks.append(dict(node=n['name'],boundary='input_A8',**metrics(a_code,captured[n['name']+'.input_q'])))
            env[n['name']]=out;gold[n['key']+'_out']=out;gold[n['key']+'_acc']=acc;gold[n['key']+'_input']=a_code
            qname=n['name']+'.output_q' if n['kind']=='conv' else n['name']
            checks.append(dict(node=n['name'],boundary='output_A8',**metrics(out,captured[qname])))
        final=np.repeat(np.repeat(env[head],4,axis=1),4,axis=2);gold['final']=final
        integer_metrics.update(torch.from_numpy(final.copy()).float().unsqueeze(0)*by[head]['scale'],y)
        if sample<4:np.savez(output/f'golden_validation_{sample}.npz',**gold)
        results.append(dict(validation_id=split['validation'][sample],qat_code_comparisons=checks,
            final_qat_comparison=metrics(final,torch.round(fp/model.get_submodule(head).output_q.scale).to(torch.int8).squeeze(0).numpy()),
            wall_seconds=time.monotonic()-started))
        print(json.dumps({'sample':sample,'validation_id':split['validation'][sample],
            'final_qat_comparison':results[-1]['final_qat_comparison'],'wall_seconds':results[-1]['wall_seconds']}),flush=True)
    for hook in hooks:hook.remove()
    assert sha(cp)==before
    report=dict(status='INTEGER_ORACLE_EXPORTED_NOT_RTL_VERIFIED',checkpoint_sha256=before,
        validation_count=count,test_accessed=False,training=False,checkpoint_modified=False,
        convs=25,adds=3,scalar_spatial_dot_checks_per_conv=2,edge_lut_tables=len(luts),
        synthetic=a.synthetic,QAT_metrics=None if a.synthetic else qat_metrics.compute(),integer_reference_metrics=None if a.synthetic else integer_metrics.compute(),
        samples=results,integer_vs_QAT_bit_exact=all(s['final_qat_comparison']['mismatches']==0 for s in results),
        full_network_hardware_cycles=None,W2_transport_RTL_verified=False)
    (output/'reference_report.json').write_text(json.dumps(report,indent=2)+'\n')



if __name__=='__main__':main()
