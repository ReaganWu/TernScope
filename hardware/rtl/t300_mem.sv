`timescale 1ns/1ps
// Explicit synchronous research SRAM contracts; mapping/macros/timing remain separate.
module t300_sram_1rw #(parameter W=128,A=12)(input wire clk,en,we,input wire[A-1:0]addr,input wire[W-1:0]wdata,output reg[W-1:0]rdata);
 reg[W-1:0]mem[0:(1<<A)-1];
 always @(posedge clk)if(en)begin if(we)mem[addr]<=wdata;else rdata<=mem[addr];end
endmodule
module t300_sram_1r1w #(parameter W=512,A=7)(input wire clk,re,we,input wire[A-1:0]raddr,waddr,input wire[W-1:0]wdata,output reg[W-1:0]rdata);
 reg[W-1:0]mem[0:(1<<A)-1];
 always @(posedge clk)begin if(we)mem[waddr]<=wdata;if(re)rdata<=(we&&raddr==waddr)?wdata:mem[raddr];end
endmodule
module t300_edge_bank(input wire clk,we,re,zero,input wire[8:0]waddr,raddr,input wire[7:0]wdata,output reg[7:0]rdata);
 reg[7:0]mem[0:511];
 always @(posedge clk)begin if(we)mem[waddr]<=wdata;else if(re)rdata<=zero?8'd0:mem[raddr];end
endmodule
