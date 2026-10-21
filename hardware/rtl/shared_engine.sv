`timescale 1ns/1ps
module shared_engine #(parameter GENERIC=0,CONV_ONLY=0,W=(GENERIC?65:39),MW=(GENERIC?32:(CONV_ONLY?25:26)),AW=(GENERIC?33:20))(
 input clk,reset,in_valid,out_ready,add_mode,legacy_bias,
 input [3:0] lane_mask,input [31:0] tag,
 input [127:0] operand,multiplier,input [191:0] bias,
 input [5:0] shift,input signed [7:0] minimum,maximum,
 output in_ready,output reg out_valid,output reg [31:0] result,out_tag,output reg [3:0] out_mask);
 reg v1,v2,add1;reg [3:0] mask1,mask2;reg [31:0] tag1,tag2;
 reg [5:0] shift1,shift2;reg signed [7:0] lo1,hi1,lo2,hi2;
 reg signed [W-1:0] product[0:3],combined[0:3];
 wire advance=!out_valid||out_ready;assign in_ready=advance;
 function automatic signed [W-1:0] calculate;
 input [31:0] a,m;input signed [47:0] b;input add,legacy;
 reg signed [AW-1:0] x;reg signed [MW-1:0] coeff;
 begin
 coeff=m[MW-1:0];
 if(add && !CONV_ONLY)x=$signed(a[7:0]);
 else if(GENERIC && legacy)x=$signed(a)+$signed(b[31:0]);
 else if(GENERIC)x=$signed(a);
 else x=$signed(a[19:0]);
 calculate=x*coeff+((add && !CONV_ONLY)||(GENERIC&&legacy)?48'sd0:b);
 end endfunction
 function automatic signed [7:0] quant;
 input signed [W-1:0] value;input [5:0] s;input signed [7:0] lo,hi;
 reg signed [W:0] ext,answer;reg [W:0] mag,q,r,h,mask;integer sh;reg neg;
 begin
 sh=GENERIC?s:24;ext=value;neg=ext<0;mag=neg?-ext:ext;
 q=mag>>sh;mask=sh==0?0:({{W{1'b0}},1'b1}<<sh)-1;r=mag&mask;
 h=sh==0?0:({{W{1'b0}},1'b1}<<(sh-1));q=q+((sh!=0)&&((r>h)||((r==h)&&q[0])));
 answer=neg?-$signed(q):$signed(q);
 quant=answer<lo?lo:(answer>hi?hi:answer[7:0]);
 end endfunction
 integer i;
 always @(posedge clk)begin
 if(reset)begin v1<=0;v2<=0;out_valid<=0;end
 else if(advance)begin
 v1<=in_valid;v2<=v1;out_valid<=v2;
 add1<=add_mode&&!CONV_ONLY;mask1<=add_mode&&!CONV_ONLY?(lane_mask&4'b0011):lane_mask;
 tag1<=tag;shift1<=shift;lo1<=GENERIC?minimum:-8'sd127;hi1<=GENERIC?maximum:8'sd127;
 for(i=0;i<4;i=i+1) product[i]<=calculate(operand[32*i+:32],multiplier[32*i+:32],$signed(bias[48*i+:48]),add_mode,legacy_bias);
 if(add1)begin
 combined[0]<=product[0]+product[1];combined[1]<=product[2]+product[3];combined[2]<=0;combined[3]<=0;
 end else for(i=0;i<4;i=i+1)combined[i]<=product[i];
 mask2<=mask1;tag2<=tag1;shift2<=shift1;lo2<=lo1;hi2<=hi1;
 for(i=0;i<4;i=i+1)result[8*i+:8]<=mask2[i]?quant(combined[i],shift2,lo2,hi2):8'd0;
 out_tag<=tag2;out_mask<=mask2;
 end end
endmodule
