`timescale 1ns/1ps
module tb_cdc;
 bit clk=0,mem_clk=0;always #5 clk=~clk;always #1 mem_clk=~mem_clk;
 bit reset=1;logic req_valid=0,req_ready,req_write=1;logic[31:0]req_addr;logic[15:0]req_rows;
 logic wr_valid=0,wr_ready;logic[127:0]wr_data;logic rd_valid,rd_ready=0;logic[127:0]rd_data;logic rd_last,dma_done,dma_error;
 logic m_req_valid,m_req_ready=0,m_req_write,m_wr_valid,m_wr_ready=0,m_rd_valid=0,m_rd_ready,m_rd_last=0,m_done=0,m_error=0;
 logic[31:0]m_req_addr;logic[15:0]m_req_rows;logic[127:0]m_wr_data,m_rd_data;
 ternscope_row_cdc c(.mem_reset(reset),.*);
 bit completion_sent=0;
 integer wi=0,wo=0,ri=0,ro=0,qin=0,qout=0,clockcount=0,memcount=0,donecount=0;
 function automatic [127:0]pat(input integer n);pat={32'(n+1025),32'(n*213),32'(n^32'habcdef12),32'(n^32'hf1234567)};endfunction
 always @(posedge clk)begin
 clockcount<=clockcount+1;if(clockcount>4000)$fatal(1,"timeout");
 if(!reset)begin
 if(req_valid&&req_ready)qin<=qin+1;
 if(wr_valid&&wr_ready)wi<=wi+1;
 if(rd_valid&&rd_ready)begin if(rd_data!==pat(ro))$fatal(1,"read data %0d",ro);if(rd_last!=(ro==1023))$fatal(1,"read last");ro<=ro+1;end
 if(dma_done)begin if(dma_error||ro!=1024)$fatal(1,"premature completion rows=%0d",ro);donecount<=donecount+1;end
 end end
 always @(negedge clk)if(!reset)begin
 req_valid=qin<64;req_addr=qin*16;req_rows=qin+1;
 wr_valid=wi<1024;wr_data=pat(wi);rd_ready=(clockcount%5!=0)&&(clockcount>160);
 end
 always @(posedge mem_clk)if(!reset)begin
 memcount<=memcount+1;
 if(m_req_valid&&m_req_ready)begin if(m_req_addr!=qout*16||m_req_rows!=qout+1||!m_req_write)$fatal(1,"req order");qout<=qout+1;end
 if(m_wr_valid&&m_wr_ready)begin if(m_wr_data!==pat(wo))$fatal(1,"write data");wo<=wo+1;end
 if(m_rd_valid&&m_rd_ready)ri<=ri+1;
 end
 always @(negedge mem_clk)if(!reset)begin
 m_req_ready=memcount%7!=0;m_wr_ready=memcount%11!=0;
 m_rd_valid=ri<1024;m_rd_data=pat(ri);m_rd_last=ri==1023;
 m_done=ri==1024&&!completion_sent;if(m_done)completion_sent=1;
 end
 initial begin repeat(8)@(negedge clk);reset=0;wait(qout==64&&wo==1024&&ro==1024&&donecount==1);@(negedge clk);$display("PASS CDC req64 write1024 read1024 completion1 core_cycles=%0d",clockcount);$finish;end
endmodule
