module tb_ddr;
 bit clk=0;always #5 clk=~clk;bit reset=1;
 logic req_valid=0,req_ready,req_write;logic [31:0] req_addr;logic [15:0] req_rows;
 logic wr_valid=0,wr_ready;logic[127:0]wr_data;logic rd_valid,rd_ready=0;logic[127:0]rd_data;logic rd_last,done,error;
 t300_ddr_harness dut(.*);
 integer cycles=0,wcount=0,rcount=0,startcycle;integer bursts=0;
 always @(posedge clk) begin cycles<=cycles+1;if(cycles>100000)$fatal(1,"timeout");if(dut.ar_valid&&dut.ar_ready||dut.aw_valid&&dut.aw_ready)begin bursts<=bursts+1;if(dut.ar_len>7||dut.aw_len>7)$fatal(1,"oversized DRAM transaction");end end
 function automatic [127:0] pattern(input integer n);pattern={32'(n^32'hfedcba98),32'(n*173),32'(n+91),32'(n^32'h01234567)};endfunction
 initial begin
 repeat(8)@(negedge clk);reset=0;repeat(3)@(negedge clk);
 req_addr=32'hff0;req_rows=137;req_write=1;req_valid=1;startcycle=cycles;
 do @(posedge clk);while(!req_ready);@(negedge clk);req_valid=0;
 while(wcount<137)begin wr_valid=1;wr_data=pattern(wcount);@(posedge clk);if(wr_ready)wcount++;@(negedge clk);end
 wr_valid=0;wait(done);@(negedge clk);if(error)$fatal(1,"write error");$display("WRITE rows=%0d cycles=%0d",wcount,cycles-startcycle);
 req_write=0;req_valid=1;startcycle=cycles;
 do @(posedge clk);while(!req_ready);@(negedge clk);req_valid=0;
 while(rcount<137)begin rd_ready=(cycles%7!=0);@(posedge clk);if(rd_valid&&rd_ready)begin if(rd_data!==pattern(rcount))$fatal(1,"data row %0d got %h",rcount,rd_data);if(rd_last!=(rcount==136))$fatal(1,"last");rcount++;end @(negedge clk);end
 rd_ready=0;if(error)$fatal(1,"read error");$display("PASS DRAMSIM2 roundtrip rows=%0d read_cycles=%0d bursts=%0d CLOCK_HZ=100000000",rcount,cycles-startcycle,bursts);$finish;
 end
endmodule
