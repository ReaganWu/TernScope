`timescale 1ns/1ps
module tb_affine;
  reg clk=0, reset=1, in_valid=0, out_ready=1, add_mode=0, legacy_bias=0;
  always #5 clk=~clk;
  reg [3:0] lane_mask=15;
  reg [31:0] tag=0;
  reg [127:0] operand=0, multiplier=0;
  reg [191:0] bias=0;
  reg [5:0] shift=24;
  reg signed [7:0] minimum=-127, maximum=127;
  wire in_ready, out_valid;
  wire [31:0] result, out_tag;
  wire [3:0] out_mask;
  shared_engine dut(.*);
  task check(input integer id, input [31:0] expected, input [3:0] mask);
    integer guard;
    begin
      @(negedge clk); tag=id; in_valid=1;
      @(posedge clk); if(!in_ready) $fatal(1,"affine input unexpectedly stalled");
      @(negedge clk); in_valid=0; out_ready=0;
      guard=0;
      while(!out_valid) begin @(negedge clk); guard=guard+1; if(guard>10)$fatal(1,"affine timeout");end
      repeat(3) begin
        if(result!==expected || out_tag!==id || out_mask!==mask) $fatal(1,"affine result/stall id=%0d result=%h",id,result);
        @(negedge clk);
      end
      out_ready=1; @(negedge clk);
    end
  endtask
  initial begin
    repeat(4) @(negedge clk); reset=0;
    operand={-32'sd3,-32'sd1,32'd3,32'd1};
    multiplier={4{32'd8388608}};
    check(1,32'hfe000200,15); // [-0.5,1.5] and their negative ties: even rounding.
    operand={-32'sd200,32'd200,-32'sd127,32'd127};
    multiplier={4{32'd16777216}};
    check(2,32'h817f817f,15); // Saturate signed A8 to [-127,127].
    add_mode=1;
    operand={-32'sd2,32'd3,32'd1,32'd1};
    multiplier={4{32'd8388608}};
    check(3,32'h00000001,3); // Sum products BEFORE rounding: .5+.5=1.
    $display("PASS affine: RNE ties, saturation, pairwise Add, masks, output stall");
    $finish;
  end
endmodule
