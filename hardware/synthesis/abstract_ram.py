import json,sys
j=json.load(open(sys.argv[1])); ledger=[]
for mn,m in list(j['modules'].items()):
 for cn,c in m.get('cells',{}).items():
  if c['type']!='$mem_v2' or int(c['parameters']['WR_PORTS'],2)==0:continue
  if not ('ternscope_sram_1rw' in mn or 'ternscope_sram_1r1w' in mn or 'ternscope_edge_bank' in mn or (mn=='acc_row_fifo' and cn=='memory')):continue
  p=c['parameters']; num=lambda k:int(p[k],2)
  name='ASIC_UNMAPPED_RAM_%d'%len(ledger)
  ports={k:{'direction':d,'bits':list(range(2+sum(len(v) for kk,v in list(c['connections'].items())[:i]),2+sum(len(v) for kk,v in list(c['connections'].items())[:i])+len(c['connections'][k])))} for i,(k,d) in enumerate(c['port_directions'].items())}
  j['modules'][name]={'attributes':{'blackbox':'1'},'ports':ports,'cells':{},'netnames':{}}
  ledger.append({'module':mn,'instance':cn,'macro':name,'width':num('WIDTH'),'depth':num('SIZE'),'bits':num('WIDTH')*num('SIZE'),'read_ports':num('RD_PORTS'),'write_ports':num('WR_PORTS'),'read_clock_enable':p['RD_CLK_ENABLE'],'write_clock_enable':p['WR_CLK_ENABLE'],'port_widths':{k:len(v) for k,v in c['connections'].items()},'area':None})
  c['type']=name;c['parameters']={}
json.dump(j,open(sys.argv[2],'w'))
json.dump({'memories':ledger,'total_bits':sum(x['bits'] for x in ledger),'note':'Mutable arrays retain exact collected port contracts; no physical macro area claimed. ROM contents remain mapped logic.'},open(sys.argv[3],'w'),indent=2)
