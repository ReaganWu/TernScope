`timescale 1ns/1ps
// Fixed frozen-graph accelerator. All frame arithmetic/control executes in RTL.
// Source cache fills are initially serialized before destructive feature reuse.
module ternscope_core #(parameter integer LUT_ENTRIES=5376)(input logic clk,reset,start,cold,
 input logic input_valid,output logic input_ready,input logic [127:0] input_data,
 output logic host_valid,input logic host_ready,output logic [127:0] host_data,
 output logic done,error,output logic [63:0] cycles,
 output logic debug_valid,output logic [5:0] debug_node,output logic [15:0] debug_row,output logic [127:0] debug_data,
 output logic node_done,output logic [5:0] node_done_id,
 output logic req_valid,input logic req_ready,output logic req_write,output logic [31:0] req_addr,output logic [15:0] req_rows,
 output logic wr_valid,input logic wr_ready,output logic [127:0] wr_data,
 input logic rd_valid,output logic rd_ready,input logic [127:0] rd_data,input logic rd_last,dma_done,dma_error);
 localparam NT=448;
 reg[127:0] completed_row;integer completed_logical,completed_physical;
 integer z,pa;integer add_addr0,add_addr1,addpixel0,addpixel1,add_next_addr0,add_next_addr1,add_next_pixel0,add_next_pixel1;
 reg [639:0] nodes[0:27],nr;
 reg [511:0] tiles[0:NT-1],tr;
 reg [127:0] pm[0:275];reg [191:0] pb[0:275];reg[31:0]add0[0:27],add1[0:27];
 reg [7:0] lutrom[0:LUT_ENTRIES-1];





 reg [7:0] headline[0:63];
 initial begin
 $readmemh("nodes.hex",nodes);$readmemh("tiles.hex",tiles);
 $readmemh("M128.hex",pm);$readmemh("B192.hex",pb);$readmemh("LUT8.hex",lutrom);
 $readmemh("add0.hex",add0);$readmemh("add1.hex",add1);
 end
 reg active_cache,cached_ready,pf_ready,pf_started;integer pfstate,pfd,pfi,pfbase,pfrows,pfdb,pf_qaddr,pfphysical;reg[511:0]ptr;reg pf_qvalid;
 function automatic integer pf(input integer f);pf=ptr[16*f+:16];endfunction
 function automatic integer div_small(input integer a,b);
 case(b)1:div_small=a;2:div_small=a/2;3:div_small=a/3;4:div_small=a/4;5:div_small=a/5;6:div_small=a/6;7:div_small=a/7;8:div_small=a/8;9:div_small=a/9;16:div_small=a/16;32:div_small=a/32;64:div_small=a/64;128:div_small=a/128;256:div_small=a/256;default:div_small=0;endcase
 endfunction
 function automatic integer rem_small(input integer a,b);
 case(b)1:rem_small=0;2:rem_small=a%2;3:rem_small=a%3;4:rem_small=a%4;5:rem_small=a%5;6:rem_small=a%6;7:rem_small=a%7;8:rem_small=a%8;9:rem_small=a%9;16:rem_small=a%16;32:rem_small=a%32;64:rem_small=a%64;128:rem_small=a%128;256:rem_small=a%256;default:rem_small=0;endcase
 endfunction
 function automatic integer nf(input integer f);nf=nr[16*f+:16];endfunction
 function automatic integer tf(input integer f);tf=tr[16*f+:16];endfunction
 function automatic integer cache_index(input integer src,pixel,block,logical);
 integer d,b;begin cache_index=-1;
 for(d=0;d<2;d=d+1)begin b=5+10*d;
 if(d<tf(3)&&tf(b)==src)begin
 if(src==0)begin if(logical>=tf(b+6)&&logical<tf(b+6)+tf(b+7))cache_index=tf(b+5)+logical-tf(b+6);end
 else if(pixel>=tf(b+1)&&pixel<tf(b+1)+tf(b+2)&&block>=tf(b+3)&&block<tf(b+3)+tf(b+4))cache_index=tf(b+5)+(pixel-tf(b+1))*tf(b+4)+block-tf(b+3);
 end end end endfunction
 typedef enum logic [5:0] {IDLE,COLDREQ,COLDREAD,INGREQ,INGWRITE,TILELOAD,NODELOAD,EDGELOAD,FILLSTART,FILLLOCAL,FILLLOCALEND,FILLREQ,FILLREAD,WEIGHTLOAD,STREAM,STREAMWAIT,DRAIN,STEMREQ,STEMWRITE,TILENEXT,ADDREAD0,ADDREAD1,ADDSEND,ADDWAIT,FINALREAD,FINALREQ,FINALWRITE,HOSTREQ,HOSTREAD,FINISH,ADDREAD2} state_t;
 state_t state;
 integer ti,ni,kb,nb,m,oc,fi,fd,filln,fillbase,fillphysical,fillrows;
 integer wrow,edgepos,edgecontext,edgecontexts,outcount,drain_count,commit_count,stem_count;
 integer final_y,final_count,final_word,host_count;
 wire [127:0] fq;reg fqvalid;integer fqaddr;
 wire [511:0] aq;reg [511:0]adot;reg apvalid,apclear;reg [31:0] aptag;
 wire [511:0] drain_data;reg drain_valid;
 reg [127:0] addx,addy,add_nextx,add_nexty,pack_data;
 integer addm,addgroup;
 logic fifo_inready,fifo_outvalid,fifo_outready;wire [127:0] fifo_data;wire[3:0]fifo_mask;wire[31:0]fifo_tag;wire[6:0]fifo_occupancy;
 wire pipe_reset=reset||(state==WEIGHTLOAD&&wrow==0&&kb==0);
 acc_row_fifo fifo(.clk(clk),.reset(pipe_reset),.in_valid(drain_valid&&state==DRAIN),.in_ready(fifo_inready),.in_data(drain_data),.configured_lanes(nf(13)==1?5'd1:5'd16),.out_valid(fifo_outvalid),.out_ready(fifo_outready),.out_data(fifo_data),.out_mask(fifo_mask),.out_tag(fifo_tag),.occupancy(fifo_occupancy));
 wire [31:0]wq;reg[3:0]wqt;reg wqvalid;wire[127:0]stem_q;reg stem_valid;
 logic av,ar,wv,wr;logic[127:0] aa;logic[31:0] at,wd;logic[3:0]weightrow;wire aov;wire[511:0]dot;wire[31:0]dot_tag;wire dot_clear;
 ternscope_array array(.clk(clk),.reset(reset),.weight_valid(wv),.weight_ready(wr),.weight_row(weightrow),.weight_data(wd),.in_valid(av),.in_ready(ar),.in_activation(aa),.in_tag(at),.in_clear(kb==0),.out_valid(aov),.out_ready(1'b1),.out_dot(dot),.out_tag(dot_tag),.out_clear(dot_clear));
 reg q1v,q2v,q1pad;wire[127:0]cq,q2a;reg [31:0]q1tag,q2tag;integer q1edge;
 logic engine_ready,engine_out_valid;wire[31:0] engine_result,engine_tag;wire[3:0]engine_mask;
 reg job_valid,job_mode;reg[127:0]job_a,job_m;reg[191:0]job_b;reg[31:0]job_tag;reg[3:0]job_mask;
 shared_engine engine(.clk(clk),.reset(pipe_reset),.in_valid(job_valid),.out_ready(1'b1),.add_mode(job_mode),.legacy_bias(1'b0),.lane_mask(job_mask),.tag(job_tag),.operand(job_a),.multiplier(job_m),.bias(job_b),.shift(6'd24),.minimum(-8'sd127),.maximum(8'sd127),.in_ready(engine_ready),.out_valid(engine_out_valid),.result(engine_result),.out_tag(engine_tag),.out_mask(engine_mask));
 integer src,sourceedge,cb,ky,kx,iy,ix,pixel,block,addr,logical,cn,sbase,spatial,i,j,d,drb;
 reg pad;wire feature_commit=engine_out_valid&&(nf(0)?engine_tag[2:0]==7:(nf(13)==1||engine_tag[1:0]==3));
 reg pfconflict;
 always_comb begin
 pfphysical=pf(pfdb+9)+(pf(pfdb+1)+div_small(pfi,pf(pfdb+4)))*pf(pfdb+8)+pf(pfdb+3)+rem_small(pfi,pf(pfdb+4));
 pfconflict=feature_commit&&ni!=0&&((completed_physical>>12)==(pfphysical>>12));
 end
 reg[127:0]rgb_data[0:7];integer rgb_tag[0:7];reg[7:0]rgb_valid;
 integer needrow[0:15],needbyte[0:15],hit[0:15],rgbmiss,rgbrpl,RGBreq,RGBvictim;reg RGBpending;wire[127:0]RGBq;
 reg[127:0]stem_a;reg stem_ready;
 always_comb begin
 cb=kb/(1); spatial=0;ky=0;kx=0;iy=0;ix=0;pixel=0;block=0;addr=0;logical=0;sourceedge=0;src=0;pad=1;sbase=18;
 if(nf(3)>3)begin
 cb=kb/(1);cb=rem_small(kb,nf(3)/16);spatial=div_small(kb,nf(3)/16);ky=div_small(spatial,nf(8));kx=rem_small(spatial,nf(8));
 iy=(div_small(tf(1)+m,nf(2)))*nf(9)+ky-nf(11);ix=(rem_small(tf(1)+m,nf(2)))*nf(10)+kx-nf(12);
 for(j=0;j<2;j=j+1)if(j<nf(17)&&cb>=nf(18+j*10+1)&&cb<nf(18+j*10+1)+nf(18+j*10+3))sourceedge=j;
 sbase=18+sourceedge*10;src=nf(sbase);block=nf(sbase+2)+cb-nf(sbase+1);
 pad=iy<0||ix<0||iy>=(nf(sbase+8)<<nf(sbase+5))||ix>=(nf(sbase+7)<<nf(sbase+4));
 pixel=(iy>>nf(sbase+5))*nf(sbase+7)+(ix>>nf(sbase+4));addr=cache_index(src,pixel,block,0);
 end
 stem_ready=1;stem_a=0;rgbmiss=-1;
 for(i=0;i<16;i=i+1)begin
 cn=kb*16+i;needrow[i]=-1;needbyte[i]=0;hit[i]=-1;
 if(cn<nf(4))begin
 ky=(cn/3)/3;kx=(cn/3)%3;iy=(div_small(tf(1)+m,nf(2)))*2+ky-1;ix=(rem_small(tf(1)+m,nf(2)))*2+kx-1;
 if(iy>=0&&iy<256&&ix>=0&&ix<256)begin
 logical=iy*768+ix*3+cn%3;needrow[i]=logical/16;needbyte[i]=logical%16;
 for(j=0;j<8;j=j+1)if(rgb_valid[j]&&rgb_tag[j]==needrow[i])hit[i]=j;
 if(hit[i]<0)begin stem_ready=0;if(rgbmiss<0)rgbmiss=needrow[i];end
 else stem_a[i*8+:8]=rgb_data[hit[i]][needbyte[i]*8+:8];
 end end end
 rgbrpl=-1;
 for(j=0;j<8;j=j+1)begin
 d=0;for(i=0;i<16;i=i+1)if(rgb_valid[j]&&rgb_tag[j]==needrow[i])d=1;
 if(!d&&rgbrpl<0)rgbrpl=j;
 end
 av=ni==0?(state==STREAM&&m<tf(2)&&stem_ready&&!RGBpending):q2v;
 aa=ni==0?stem_a:q2a;at=ni==0?m:q2tag;
 wv=state==WEIGHTLOAD&&wqvalid;weightrow=wqt;wd=wq;
 fifo_outready=state==DRAIN&&engine_ready;
 req_valid=0;req_write=0;req_addr=0;req_rows=0;wr_valid=0;wr_data=0;rd_ready=0;input_ready=0;host_valid=0;host_data=rd_data;
 case(state)
 COLDREQ:begin req_valid=1;req_addr=1048576;req_rows=3880;end
 COLDREAD:rd_ready=1;
 INGREQ:begin req_valid=1;req_write=1;req_rows=12288;end
 INGWRITE:begin wr_valid=input_valid;wr_data=input_data;input_ready=wr_ready;end
 FILLREQ:begin req_valid=1;req_addr=(tf(drb)==0?0:196608)+(tf(drb)==0?tf(drb+6):tf(drb+1))*16;req_rows=fillrows;end
 FILLREAD:rd_ready=1;
 STEMREQ:begin req_valid=pfstate==0;req_write=1;req_addr=196608+tf(1)*16;req_rows=tf(2);end
 STEMWRITE:begin wr_valid=stem_valid;wr_data=stem_q;end
 FINALREQ:begin req_valid=1;req_write=1;req_addr=458752+final_y*1024;req_rows=64;end
 FINALWRITE:begin wr_valid=final_word<64;
 for(i=0;i<16;i=i+1)wr_data[8*i+:8]=headline[((final_word%16)*16+i)/4];end
 HOSTREQ:begin req_valid=1;req_addr=458752;req_rows=4096;end
 HOSTREAD:begin rd_ready=host_ready;host_valid=rd_valid;end
 default:;
 endcase
 if(pfstate==3)begin req_valid=1;req_write=0;req_addr=(pf(pfdb)==0?0:196608)+(pf(pfdb)==0?pf(pfdb+6):pf(pfdb+1))*16;req_rows=pfrows;end
 if(pfstate==4)rd_ready=1;
 end
 wire[7:0]edge_write_data=lutrom[nf(18+10*edgecontext+6)*256+edgepos];
 genvar el;
 generate for(el=0;el<16;el=el+1)begin:edges
 ternscope_edge_bank eb(.clk(clk),.we(state==EDGELOAD),.re(q1v),.zero(q1pad),.waddr(edgecontext*256+edgepos),.raddr(q1edge*256+cq[el*8+:8]),.wdata(edge_write_data),.rdata(q2a[el*8+:8]));
 end endgenerate
 always_comb begin
 completed_row=pack_data;
 if(nf(0))completed_row[engine_tag[2:0]*16+:16]=engine_result[15:0];
 else if(nf(13)==1)begin completed_row=0;completed_row[7:0]=engine_result[7:0];end
 else completed_row[engine_tag[1:0]*32+:32]=engine_result;
 completed_logical=(tf(1)+(engine_tag>>(nf(0)?3:2)))*nf(6)+nb;
 completed_physical=nf(14)+completed_logical;
 end
 logic feature_re;integer feature_addr;wire[127:0]feature_q[0:2];reg[1:0]feature_selected;
 always_comb begin
 feature_re=0;feature_addr=0;
 if(pfstate==2&&!pfconflict)begin feature_re=1;feature_addr=pfphysical;end
 else if(state==FILLLOCAL)begin feature_re=1;feature_addr=tf(drb+9)+(tf(drb+1)+div_small(fi,tf(drb+4)))*tf(drb+8)+tf(drb+3)+rem_small(fi,tf(drb+4));end
 else if(state==FINALREAD&&final_count<64)begin feature_re=1;feature_addr=nf(14)+final_y*64+final_count;end
 end
 always @(posedge clk)if(feature_re)feature_selected<=feature_addr>>12;
 assign fq=feature_q[feature_selected];
 genvar fb;
 generate for(fb=0;fb<3;fb=fb+1)begin:feature_banks
 wire we=feature_commit&&ni!=0&&(completed_physical>>12)==fb;
 wire re=feature_re&&(feature_addr>>12)==fb;
 ternscope_sram_1rw #(.W(128),.A(12))ram(.clk(clk),.en(we||re),.we(we),.addr(we?completed_physical[11:0]:feature_addr[11:0]),.wdata(completed_row),.rdata(feature_q[fb]));
 end endgenerate
 logic cache_re,cache_active_we,cache_inactive_we;integer cache_addr,cache_active_wa,cache_inactive_wa;logic[127:0]cache_active_wd,cache_inactive_wd;wire[127:0]cache_q[0:1];
 always_comb begin
 cache_re=0;cache_addr=0;cache_active_we=0;cache_inactive_we=0;cache_active_wa=0;cache_inactive_wa=0;cache_active_wd=0;cache_inactive_wd=0;
 if(state==STREAM&&ni!=0&&m<tf(2)&&!pad)begin cache_re=1;cache_addr=addr;end
 else if(state==STREAM&&ni==0&&!RGBpending&&!stem_ready&&m<tf(2))begin cache_re=1;cache_addr=cache_index(0,0,0,rgbmiss);end
 else if(state==ADDREAD0)begin cache_re=1;cache_addr=add_addr0;end
 else if(state==ADDREAD1)begin cache_re=1;cache_addr=add_addr1;end
 else if(state==ADDSEND&&addm+1<tf(2)&&engine_ready&&addgroup<2)begin cache_re=1;cache_addr=addgroup==0?add_next_addr0:add_next_addr1;end
 if(state==FILLREAD&&rd_valid)begin cache_active_we=1;cache_active_wa=fillbase+fi;cache_active_wd=rd_data;end
 else if((state==FILLLOCAL||state==FILLLOCALEND)&&fqvalid)begin cache_active_we=1;cache_active_wa=fqaddr;cache_active_wd=fq;end
 if(pf_qvalid)begin cache_inactive_we=1;cache_inactive_wa=pf_qaddr;cache_inactive_wd=fq;end
 else if(pfstate==4&&rd_valid)begin cache_inactive_we=1;cache_inactive_wa=pfbase+pfi;cache_inactive_wd=rd_data;end
 end
 assign cq=cache_q[active_cache];assign RGBq=cq;
 genvar cbk;
 generate for(cbk=0;cbk<2;cbk=cbk+1)begin:source_cache_banks
 wire we=active_cache==cbk?cache_active_we:cache_inactive_we;
 wire re=active_cache==cbk&&cache_re;
 wire[10:0]wa=active_cache==cbk?cache_active_wa:cache_inactive_wa;
 wire[127:0]data=active_cache==cbk?cache_active_wd:cache_inactive_wd;
 ternscope_sram_1rw #(.W(128),.A(11))ram(.clk(clk),.en(we||re),.we(we),.addr(we?wa:cache_addr[10:0]),.wdata(data),.rdata(cache_q[cbk]));
 end endgenerate
 wire acc_drain_read=state==DRAIN&&(!drain_valid||fifo_inready)&&drain_count<tf(2);
 wire[511:0]acc_data;wire[511:0]acc_write_data;
 genvar ac;
 generate for(ac=0;ac<16;ac=ac+1)begin:acc_add
 assign acc_write_data[ac*32+:32]=apclear?adot[ac*32+:32]:$signed(aq[ac*32+:32])+$signed(adot[ac*32+:32]);
 end endgenerate
 ternscope_sram_1r1w #(.W(512),.A(7))acc_ram(.clk(clk),.re(aov||acc_drain_read),.we(apvalid),.raddr(aov?dot_tag[6:0]:drain_count[6:0]),.waddr(aptag[6:0]),.wdata(acc_write_data),.rdata(acc_data));
 assign aq=acc_data;assign drain_data=acc_data;
 wire weight_re=state==WEIGHTLOAD&&(!wqvalid||wr)&&wrow<16;
 wire[127:0]weight_q;
 ternscope_sram_1rw #(.W(128),.A(12))weight_ram(.clk(clk),.en(weight_re||(state==COLDREAD&&rd_valid)),.we(state==COLDREAD&&rd_valid),.addr(state==COLDREAD?fi[11:0]:((nf(15)/16)+(kb*nf(6)+nb)*4+wrow/4)),.wdata(rd_data),.rdata(weight_q));
 assign wq=weight_q[wqt[1:0]*32+:32];
 wire stem_re=state==STEMWRITE&&(!stem_valid||wr_ready)&&stem_count<tf(2);
 wire stem_we=feature_commit&&ni==0;
 ternscope_sram_1rw #(.W(128),.A(8))stem_ram(.clk(clk),.en(stem_re||stem_we),.we(stem_we),.addr(stem_we?(engine_tag>>2):stem_count[7:0]),.wdata(completed_row),.rdata(stem_q));
 always_comb begin
 addpixel0=((div_small(tf(1)+addm,nf(2)))>>nf(23))*nf(25)+((rem_small(tf(1)+addm,nf(2)))>>nf(22));
 addpixel1=((div_small(tf(1)+addm,nf(2)))>>nf(33))*nf(35)+((rem_small(tf(1)+addm,nf(2)))>>nf(32));
 add_next_pixel0=((div_small(tf(1)+addm+1,nf(2)))>>nf(23))*nf(25)+((rem_small(tf(1)+addm+1,nf(2)))>>nf(22));
 add_next_pixel1=((div_small(tf(1)+addm+1,nf(2)))>>nf(33))*nf(35)+((rem_small(tf(1)+addm+1,nf(2)))>>nf(32));
 add_next_addr0=cache_index(nf(18),add_next_pixel0,nf(20),0);add_next_addr1=cache_index(nf(28),add_next_pixel1,nf(30),0);
 add_addr0=cache_index(nf(18),addpixel0,nf(20),0);add_addr1=cache_index(nf(28),addpixel1,nf(30),0);
 end
 always_ff @(posedge clk)begin
 done<=0;debug_valid<=0;node_done<=0;job_valid<=0;
 if(reset)begin
 state<=IDLE;error<=0;cycles<=0;active_cache<=0;cached_ready<=0;pfstate<=0;pf_ready<=0;pf_started<=0;pf_qvalid<=0;ptr<=0;pfdb<=5;pfi<=0;pfd<=0;ni<=0;nr<=0;tr<=0;ti<=0;fd<=0;kb<=0;nb<=0;m<=0;wrow<=0;wqvalid<=0;stem_valid<=0;q1v<=0;q2v<=0;apvalid<=0;drain_valid<=0;rgb_valid<=0;RGBpending<=0;job_valid<=0;edgecontext<=0;edgepos<=0;edgecontexts<=1;drb<=5;addm<=0;final_y<=0;
 end else begin
 if(state!=IDLE&&state!=FINISH)cycles<=cycles+1;
 if(dma_error)error<=1;
 pf_qvalid<=0;
 if(!pf_started&&ti+1<NT&&tiles[ti+1][15:0]==ni&&((state==WEIGHTLOAD&&kb==0&&nb==0&&wrow==0)||(state==ADDREAD0&&addm==0)))begin
 ptr<=tiles[ti+1];pfd<=0;pfi<=0;pf_ready<=0;pf_started<=1;pfstate<=1;
 end

 case(pfstate)
 1:begin
 if(pfd==pf(3))begin pf_ready<=1;pfstate<=0;end
 else begin pfdb<=5+10*pfd;pfbase<=pf(5+10*pfd+5);pfrows<=pf(5+10*pfd)==0?pf(5+10*pfd+7):pf(5+10*pfd+2)*pf(5+10*pfd+4);pfi<=0;pfstate<=pf(5+10*pfd)<=1?3:2;end
 end
 2:if(!pfconflict)begin pf_qaddr<=pfbase+pfi;pf_qvalid<=1;if(pfi+1==pfrows)pfstate<=5;else pfi<=pfi+1;end
 5:begin pfd<=pfd+1;pfstate<=1;end
 3:if(req_ready)begin pfi<=0;pfstate<=4;end
 4:begin
 if(rd_valid)pfi<=pfi+1;
 if(dma_done)begin pfd<=pfd+1;pfstate<=1;end
 end
 default:;
 endcase
 q1v<=0;q2v<=q1v;q2tag<=q1tag;

 apvalid<=aov;
 if(aov)begin adot<=dot;aptag<=dot_tag;apclear<=dot_clear;end
 if(apvalid)outcount<=outcount+1;
 // One-cycle synchronous parameter ROM capture before shared engine.
 if(fifo_outvalid&&fifo_outready)begin
 job_valid<=1;job_mode<=0;job_a<=fifo_data;job_tag<=fifo_tag;job_mask<=fifo_mask;
 job_m<=pm[nf(16)/4+nb*4+fifo_tag[1:0]];job_b<=pb[nf(16)/4+nb*4+fifo_tag[1:0]];
 end
 if(engine_out_valid)begin
 if(nf(0))pack_data[engine_tag[2:0]*16+:16]<=engine_result[15:0];else pack_data[engine_tag[1:0]*32+:32]<=engine_result;
 if(nf(0)?engine_tag[2:0]==7:(nf(13)==1||engine_tag[1:0]==3))begin
 debug_valid<=1;debug_node<=ni+1;debug_row<=completed_logical;debug_data<=completed_row;commit_count<=commit_count+1;
 if(ni!=0&&completed_physical>=12288)error<=1;
 end
 end
 case(state)
 IDLE:if(start)begin cycles<=0;error<=0;ti<=0;fi<=0;rgb_valid<=0;active_cache<=0;cached_ready<=0;pfstate<=0;pf_ready<=0;pf_started<=0;state<=cold?COLDREQ:INGREQ;end
 COLDREQ:if(req_ready)begin fi<=0;state<=COLDREAD;end
 COLDREAD:begin if(rd_valid)begin fi<=fi+1;end if(dma_done)state<=INGREQ;end
 INGREQ:if(req_ready)state<=INGWRITE;
 INGWRITE:if(dma_done)state<=TILELOAD;
 TILELOAD:begin tr<=tiles[ti];state<=NODELOAD;end
 NODELOAD:begin
 ni<=tf(0);nr<=nodes[tf(0)];kb<=0;nb<=0;fd<=0;state<=FILLSTART;
 // Same node: active edge tables preserved. First tile loads requiredcontexts.
 if(ti==0||tf(0)!=ni)begin state<=EDGELOAD;edgecontext<=0;edgepos<=0;end
 end
 EDGELOAD:begin

 if(edgepos==255)begin edgepos<=0;if(edgecontext+1<nf(17))edgecontext<=edgecontext+1;else state<=FILLSTART;end else edgepos<=edgepos+1;
 end
 FILLSTART:begin
 if(cached_ready||fd==tf(3))begin cached_ready<=0;state<=nf(0)?ADDREAD0:WEIGHTLOAD;wrow<=0;outcount<=0;m<=0;addm<=0;addgroup<=0;commit_count<=0;pack_data<=0;end
 else begin drb<=5+fd*10;fi<=0;fqvalid<=0;fillbase<=tf(5+fd*10+5);fillrows<=tf(5+fd*10)==0?tf(5+fd*10+7):tf(5+fd*10+2)*tf(5+fd*10+4);state<=tf(5+fd*10)<=1?FILLREQ:FILLLOCAL;end
 end
 FILLREQ:if(req_ready)begin fi<=0;state<=FILLREAD;end
 FILLREAD:begin if(rd_valid)begin fi<=fi+1;end if(dma_done)begin fd<=fd+1;state<=FILLSTART;end end
 FILLLOCAL:begin
 
 fqaddr<=fillbase+fi;fqvalid<=1;
 if(fi+1==fillrows)state<=FILLLOCALEND;else fi<=fi+1;
 end
 FILLLOCALEND:begin fqvalid<=0;fd<=fd+1;state<=FILLSTART;end
 WEIGHTLOAD:begin
 if(!wqvalid||wr)begin
 if(wrow<16)begin wqt<=wrow;wqvalid<=1;wrow<=wrow+1;end else wqvalid<=0;
 end
 if(wqvalid&&wr&&wqt==15)begin m<=0;outcount<=0;state<=STREAM;RGBpending<=0;end
 end
 STREAM:begin
 if(ni==0)begin
 if(RGBpending)begin rgb_data[RGBvictim]<=RGBq;rgb_tag[RGBvictim]<=RGBreq;rgb_valid[RGBvictim]<=1;RGBpending<=0;end
 else if(m<tf(2)&&!stem_ready)begin
 if(rgbrpl<0||cache_index(0,0,0,rgbmiss)<0)begin error<=1;state<=FINISH;end
 else begin RGBreq<=rgbmiss;RGBvictim<=rgbrpl;RGBpending<=1;end
 end
 if(av&&ar)begin if(m+1==tf(2))state<=STREAMWAIT;else m<=m+1;end
 end else begin
 if(m<tf(2))begin
 q1v<=1;q1tag<=m;q1pad<=pad;q1edge<=sourceedge;
 if(!pad&&(addr<0||addr>=2048))begin error<=1;state<=FINISH;end
 if(m+1==tf(2))state<=STREAMWAIT;else m<=m+1;
 end end
 end
 STREAMWAIT:if(outcount==tf(2))begin
 if(kb+1<nf(5))begin kb<=kb+1;wrow<=0;state<=WEIGHTLOAD;end
 else begin state<=DRAIN;drain_count<=0;drain_valid<=0;commit_count<=0;pack_data<=0;end
 end
 DRAIN:begin
 if(!drain_valid||fifo_inready)begin
 if(drain_count<tf(2))begin drain_valid<=1;drain_count<=drain_count+1;end else drain_valid<=0;
 end
 if(commit_count==tf(2))begin drain_valid<=0;
 if(ni==0)begin state<=STEMREQ;stem_count<=0;end
 else if(nb+1<nf(6))begin nb<=nb+1;kb<=0;wrow<=0;m<=0;state<=WEIGHTLOAD;end
 else state<=TILENEXT;
 end end
 STEMREQ:if(req_ready&&pfstate==0)begin stem_count<=0;stem_valid<=0;state<=STEMWRITE;end
 STEMWRITE:begin
 if(!stem_valid||wr_ready)begin
 if(stem_count<tf(2))begin stem_valid<=1;stem_count<=stem_count+1;end else stem_valid<=0;
 end
 if(dma_done)state<=TILENEXT;
 end
 ADDREAD0:state<=ADDREAD1;
 ADDREAD1:begin addx<=cq;state<=ADDREAD2;end
 ADDREAD2:begin addy<=cq;state<=ADDSEND;end
 ADDSEND:if(engine_ready)begin
 if(addm+1<tf(2))begin
 if(addgroup==1)add_nextx<=cq;if(addgroup==2)add_nexty<=cq;
 end
 job_valid<=1;job_mode<=1;job_mask<=3;job_tag<=(addm<<3)|addgroup;job_b<=0;
 job_a[31:0]<={{24{addx[addgroup*16+7]}},addx[addgroup*16+:8]};job_a[63:32]<={{24{addy[addgroup*16+7]}},addy[addgroup*16+:8]};
 job_a[95:64]<={{24{addx[addgroup*16+15]}},addx[addgroup*16+8+:8]};job_a[127:96]<={{24{addy[addgroup*16+15]}},addy[addgroup*16+8+:8]};
 job_m<={add1[ni],add0[ni],add1[ni],add0[ni]};
 if(addgroup==7)begin addgroup<=0;if(addm+1<tf(2))begin addm<=addm+1;addx<=add_nextx;addy<=add_nexty;end else state<=ADDWAIT;end else addgroup<=addgroup+1;
 end
 ADDWAIT:if(commit_count==tf(2))state<=TILENEXT;
 TILENEXT:begin
 if(ti+1==NT)begin node_done<=1;node_done_id<=28;final_y<=0;final_count<=0;fqvalid<=0;state<=FINALREAD;end
 else if(tiles[ti+1][15:0]!=ni||pf_ready)begin
 if(tiles[ti+1][15:0]!=ni)begin node_done<=1;node_done_id<=ni+1;cached_ready<=0;end
 else begin active_cache<=!active_cache;cached_ready<=1;end
 pf_ready<=0;pf_started<=0;ti<=ti+1;state<=TILELOAD;
 end
 end
 FINALREAD:begin
 if(fqvalid)headline[fqaddr]<=fq[7:0];
 if(final_count<64)begin fqaddr<=final_count;fqvalid<=1;final_count<=final_count+1;end
 else begin fqvalid<=0;final_word<=0;state<=FINALREQ;end
 end
 FINALREQ:if(req_ready)begin final_word<=0;state<=FINALWRITE;end
 FINALWRITE:begin if(wr_ready&&wr_valid)final_word<=final_word+1;if(dma_done)begin if(final_y==63)state<=HOSTREQ;else begin final_y<=final_y+1;final_count<=0;fqvalid<=0;state<=FINALREAD;end end end
 HOSTREQ:if(req_ready)state<=HOSTREAD;
 HOSTREAD:if(dma_done)state<=FINISH;
 FINISH:begin done<=1;state<=IDLE;end
 default:state<=FINISH;
 endcase
 end end
endmodule
