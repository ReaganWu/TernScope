"""Combine fixed T300 geometry with the selected checkpoint's numeric parameters."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    graph=json.loads((args.artifacts/'graph.json').read_text())
    parameters=np.load(args.artifacts/'parameters.npz')
    templates=Path(__file__).resolve().parents[1]/'control/geometry'
    words=[int(line,16) for line in (templates/'nodes.hex').read_text().splitlines()]
    if len(graph['nodes'])!=28 or len(words)!=28:
        parser.error('Only the frozen 28-node TernScope topology is supported.')
    nodes=[];multipliers=[];biases=[];add0=[];add1=[];weight_base=0
    for template,node in zip(words,graph['nodes']):
        fields=[(template>>(16*i))&65535 for i in range(40)]
        shape=[int(node['kind']=='add'),*node['shape'][1:],node['ci'],node['K'],node['kb'],node['nb'],*node['kernel'],*node['stride'],*node['padding'],node['shape'][0]]
        if fields[:14]!=shape or fields[15]!=weight_base or fields[16]!=len(multipliers) or fields[17]!=len(node['sources']):
            parser.error(f'Geometry differs at {node["name"]}; regenerate the tiling schedule for a new topology.')
        for index,source in enumerate(node['sources']):
            fields[18+10*index+6]=source['lut']
        nodes.append(sum(v<<(16*i) for i,v in enumerate(fields)))
        m=parameters[node['key']+'_M'];b=parameters[node['key']+'_B']
        if np.any(m<0) or np.any(m>=(1<<25)) or np.any(np.abs(b)>=(1<<38)):
            parser.error(f'Numeric parameters exceed the bounded T300 engine at {node["name"]}.')
        if node['kind']=='conv':
            weights=parameters[node['key']+'_W'].astype(np.int64)
            bound=127*np.abs(weights).sum((1,2,3))
            channels=len(bound)
            if np.any(bound>=(1<<19)) or np.any(bound*m[:channels]+np.abs(b[:channels])>=(1<<38)):
                parser.error(f'Worst-case ternary accumulation exceeds the bounded T300 engine at {node["name"]}.')
        multipliers.extend(m.tolist());biases.extend(b.tolist())
        coeff=node.get('add_coefficients',[0,0])
        if node['kind']=='add' and (node['add_shift']!=24 or max(coeff)>=(1<<25)):
            parser.error('Add coefficients require a different bounded-engine configuration.')
        add0.append(coeff[0]);add1.append(coeff[1])
        weight_base+=parameters[node['key']+'_W2'].nbytes
    if len(multipliers)!=1104 or weight_base!=62080:
        parser.error('Packed parameter dimensions differ from the T300 layout.')
    # Check the actual independent-oracle sample arithmetic before narrowing it.
    for golden in sorted(args.artifacts.glob('golden_validation_*.npz')):
        with np.load(golden) as sample:
            for node in graph['nodes']:
                if node['kind']!='conv':continue
                acc=sample[node['key']+'_acc'].astype(np.int64)
                count=node['shape'][0]
                value=acc*parameters[node['key']+'_M'][:count,None,None]+parameters[node['key']+'_B'][:count,None,None]
                if np.any(np.abs(acc)>=(1<<19)) or np.any(value<-(1<<38)) or np.any(value>=(1<<38)):
                    parser.error(f'Sample arithmetic exceeds bounded-engine widths at {node["name"]}.')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'nodes.hex').write_text(''.join(f'{v:0160x}\n' for v in nodes))
    shutil.copyfile(templates/'tiles.hex',args.output/'tiles.hex')
    def pack(name,values,bits):
        packed=[sum((int(v)&((1<<bits)-1))<<(bits*j) for j,v in enumerate(values[i:i+4])) for i in range(0,len(values),4)]
        (args.output/name).write_text(''.join(f'{v:0{bits}x}\n' for v in packed))
    pack('M128.hex',multipliers,32);pack('B192.hex',biases,48)
    for name,values in [('add0.hex',add0),('add1.hex',add1)]:
        (args.output/name).write_text(''.join(f'{int(v)&0xffffffff:08x}\n' for v in values))
    lut=parameters['edge_LUTs'].astype(np.int8).view(np.uint8).ravel()
    (args.output/'LUT8.hex').write_text(''.join(f'{int(v):02x}\n' for v in lut))
    (args.output/'config.json').write_text(json.dumps(dict(lut_entries=len(lut),lut_tables=graph['lut_tables'],nodes=28,tiles=448,weight_bytes=weight_base),indent=2)+'\n')
    print(f'Control generated: {graph["lut_tables"]} LUT tables, {len(lut)} entries. Use this directory with the simulation builder.')


if __name__=='__main__':
    main()
