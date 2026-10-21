`timescale 1ns/1ps
// Dual-clock, Gray pointer CDC FIFO; two synchronizer stages per pointer.
module async_fifo #(parameter WIDTH=128, LGDEPTH=4)(
 input logic wclk,wreset,wvalid,output logic wready,input logic[WIDTH-1:0] wdata,
 input logic rclk,rreset,output logic rvalid,input logic rready,output logic[WIDTH-1:0] rdata);
 localparam P=LGDEPTH+1;
 logic[WIDTH-1:0]mem[0:(1<<LGDEPTH)-1];
 logic[P-1:0]wb, wg,rb,rg;
 (* ASYNC_REG="TRUE" *) logic[P-1:0]rg1,rg2,wg1,wg2;
 logic[P-1:0] wn,wgn,rn;
 logic full;
 assign wready=!full;
 assign wn=wb+(wvalid&&wready);assign wgn=(wn>>1)^wn;
 assign rn=rb+1;
 assign rvalid=rg!=wg2;
 assign rdata=mem[rb[LGDEPTH-1:0]];
 always_ff @(posedge wclk) begin
 if(wreset)begin wb<=0;wg<=0;rg1<=0;rg2<=0;full<=0;end
 else begin
 rg1<=rg;rg2<=rg1;wb<=wn;wg<=wgn;
 full<=wgn=={~rg2[P-1:P-2],rg2[P-3:0]};
 if(wvalid&&wready)mem[wb[LGDEPTH-1:0]]<=wdata;
 end end
 always_ff @(posedge rclk)begin
 if(rreset)begin rb<=0;rg<=0;wg1<=0;wg2<=0;end
 else begin
 wg1<=wg;wg2<=wg1;
 if(rvalid&&rready)begin rb<=rn;rg<=(rn>>1)^rn;end
 end end
endmodule
