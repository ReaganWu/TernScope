`timescale 1ns/1ps
module acc_row_fifo(input clk,reset,in_valid,out_ready,input [511:0] in_data,input [4:0] configured_lanes,
 output in_ready,out_valid,output [127:0] out_data,output [3:0] out_mask,output [31:0] out_tag,output [6:0] occupancy);
 reg [511:0] memory[0:63];reg [511:0] head;
 reg [5:0] wr,rd;reg [6:0] count;reg [1:0] group;reg [29:0] row_sequence;
 wire last=(configured_lanes==1)||(group==3);wire pop=out_valid&&out_ready&&last;
 assign in_ready=(count<64)||pop;wire push=in_valid&&in_ready;
 assign out_valid=count!=0;assign occupancy=count;
 assign out_data=head>>(group*128);assign out_mask=configured_lanes==1?4'b0001:4'b1111;
 assign out_tag={row_sequence,group};
 always @(posedge clk)begin
 if(reset)begin wr<=0;rd<=0;count<=0;group<=0;row_sequence<=0;end
 else begin
 if(push)begin memory[wr]<=in_data;wr<=wr+1'b1;end
 if(push&&!pop)count<=count+1'b1;else if(pop&&!push)count<=count-1'b1;
 if(out_valid&&out_ready)begin
 if(last)begin rd<=rd+1'b1;group<=0;row_sequence<=row_sequence+1'b1;end else group<=group+1'b1;
 end
 // Registered empty/one-row forwarding; synchronous read otherwise, no same-address RAM assumption.
 if(count==0&&push)head<=in_data;
 else if(pop)begin
 if(count==1&&push)head<=in_data;
 else if(count>1)head<=memory[(rd+1'b1)&6'h3f];
 end
 end end
endmodule
