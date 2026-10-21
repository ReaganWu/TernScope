`timescale 1ns/1ps
module tb_array;
  reg clk=0, reset=1;
  always #5 clk=~clk;
  reg weight_valid=0, in_valid=0, out_ready=0, in_clear=0;
  wire weight_ready, in_ready, out_valid, out_clear;
  reg [3:0] weight_row=0;
  reg [31:0] weight_data=0, in_tag=0;
  reg [127:0] in_activation=0;
  wire [511:0] out_dot;
  wire [31:0] out_tag;
  integer sent=0, received=0, cycles=0, row, col, k, expected;
  reg sending=0;
  t300_array13 dut(.*);
  function integer code(input integer r,c);
    if(c==0) code=2;
    else if(c==1) code=1;
    else if(c==15) code=3;
    else code=(r+c)%3;
  endfunction
  function integer activation(input integer n,lane);
    if(n==0) activation=-128;
    else if(n==1) activation=127;
    else activation=((n*37+lane*19)%256)-128;
  endfunction
  always @(negedge clk) if(sending) begin
    in_valid=sent<8; in_tag=sent; in_clear=sent%2;
    for(integer lane=0;lane<16;lane=lane+1)
      in_activation[8*lane+:8]=activation(sent,lane);
    out_ready=(cycles%5)!=0;
  end
  always @(posedge clk) if(!reset) begin
    cycles=cycles+1;
    if(cycles>400) $fatal(1,"array timeout");
    if(in_valid&&in_ready) sent=sent+1;
    if(out_valid&&out_ready) begin
      if(out_tag!==received || out_clear!==(received%2==1)) $fatal(1,"array tag/order");
      for(integer c=0;c<16;c=c+1) begin
        expected=0;
        for(integer r=0;r<16;r=r+1)
          expected=expected+activation(received,r)*(code(r,c)==1?1:(code(r,c)==2?-1:0));
        if($signed(out_dot[c*32+:32])!==expected) $fatal(1,"array arithmetic n=%0d column=%0d",received,c);
      end
      received=received+1;
    end
  end
  initial begin
    repeat(4) @(negedge clk); reset=0;
    for(row=0;row<16;row=row+1) begin
      @(negedge clk); weight_valid=1; weight_row=row;
      for(col=0;col<16;col=col+1) weight_data[col*2+:2]=code(row,col);
      @(posedge clk); if(!weight_ready) $fatal(1,"weight load stalled unexpectedly");
    end
    @(negedge clk); weight_valid=0; sending=1;
    wait(received==8); @(negedge clk);
    $display("PASS array: 128 dot products, signed extrema, reserved codes, tags, backpressure");
    $finish;
  end
endmodule
