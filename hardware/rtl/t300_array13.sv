`timescale 1ns/1ps
// Fixed16-term partialsum bounds [-2048,+2048]; signed13 stores +2048.
// ABI stays signed32 via sign-extension. Not a general accumulated32-bit PE.
// W2 encoding: 00=zero, 01=+1, 10=-1; 11 reserved (treated zero).
// One independent 16-term dot product per column per accepted vector.
// Each of 256 PEs registers its running sum; rows form 16 real stages.
module t300_ternary_pe13(input wire clk, input wire reset, input wire ce,
 input wire signed [7:0] activation, input wire [1:0] weight,
 input wire signed [12:0] sum_in, output reg signed [12:0] sum_out);
 wire signed [12:0] extended = {{5{activation[7]}},activation};
 always @(posedge clk) if(reset) sum_out <= 0; else if(ce)
 case(weight)
 2'b01: sum_out <= sum_in + extended;
 2'b10: sum_out <= sum_in - extended;
 default: sum_out <= sum_in;
 endcase
endmodule

module t300_array13(input wire clk,input wire reset,
 input wire weight_valid, output wire weight_ready,
 input wire [3:0] weight_row,input wire [31:0] weight_data,
 input wire in_valid,output wire in_ready,input wire [127:0] in_activation,
 input wire [31:0] in_tag,input wire in_clear,
 output wire out_valid,input wire out_ready,output wire [511:0] out_dot,
 output wire [31:0] out_tag,output wire out_clear);
 reg [31:0] weights [0:15];
 reg [15:0] valid;
 reg [31:0] tags [0:15];
 reg [15:0] clears;
 wire ce = !valid[15] || out_ready;
 wire empty = !(|valid);
 assign weight_ready = empty && !in_valid;
 assign in_ready = ce && !weight_valid;
 assign out_valid = valid[15];
 assign out_tag = tags[15];
 assign out_clear = clears[15];
 wire [207:0] sums [0:15];
 integer i;
 always @(posedge clk) begin
 if(reset) begin
 valid <= 0; clears <= 0;
 for(i=0;i<16;i=i+1) begin weights[i]<=0; tags[i]<=0; end
 end else begin
 if(weight_valid && weight_ready) weights[weight_row] <= weight_data;
 if(ce) begin
 valid <= {valid[14:0],in_valid && in_ready};
 clears <= {clears[14:0],in_clear};
 tags[0] <= in_tag;
 for(i=1;i<16;i=i+1) tags[i] <= tags[i-1];
 end
 end
 end
 genvar r,c;
 // Triangular activation forwarding: stage r retains lanes r+1 .. 15.
 generate for(r=0;r<16;r=r+1) begin: pipe
 wire signed [7:0] a;
 if(r==0) begin: ingress
 assign a=in_activation[7:0];
 end else begin: forwarded
 assign a=pipe[r-1].suffix[7:0];
 end
 if(r<15) begin: tail
 reg [119-r*8:0] storage;
 if(r==0) begin: input_tail
 always @(posedge clk) if(reset) storage<=0; else if(ce) storage <= in_activation[127:8];
 end else begin: next_tail
 always @(posedge clk) if(reset) storage<=0; else if(ce) storage <= pipe[r-1].suffix[127-r*8:8];
 end
 wire [119-r*8:0] value=storage;
 end
 // Last row has a dummy 8-bit suffix, unused.
 wire [(r<15 ? 120-r*8 : 8)-1:0] suffix;
 if(r<15) assign suffix=tail.value; else assign suffix=0;
 for(c=0;c<16;c=c+1) begin: columns
 wire signed [12:0] prior;
 if(r==0) assign prior=0; else assign prior=sums[r-1][c*13+:13];
 t300_ternary_pe13 pe(clk,reset,ce,a,weights[r][c*2+:2],prior,sums[r][c*13+:13]);
 end
 end endgenerate
 generate for(genvar o=0;o<16;o=o+1) begin: outputs
 assign out_dot[o*32+:32]={{19{sums[15][o*13+12]}},sums[15][o*13+:13]};
 end endgenerate
endmodule
