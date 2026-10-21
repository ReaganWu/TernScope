`timescale 1ns/1ps
module t300_row_cdc(input logic clk,reset,mem_clk,mem_reset,
 input logic req_valid,output logic req_ready,input logic req_write,input logic[31:0]req_addr,input logic[15:0]req_rows,
 input logic wr_valid,output logic wr_ready,input logic[127:0]wr_data,
 output logic rd_valid,input logic rd_ready,output logic[127:0]rd_data,output logic rd_last,
 output logic dma_done,dma_error,
 output logic m_req_valid,input logic m_req_ready,output logic m_req_write,output logic[31:0]m_req_addr,output logic[15:0]m_req_rows,
 output logic m_wr_valid,input logic m_wr_ready,output logic[127:0]m_wr_data,
 input logic m_rd_valid,output logic m_rd_ready,input logic[127:0]m_rd_data,input logic m_rd_last,m_done,m_error);
 logic[48:0]rq;logic[128:0]readq;logic cv,cr,cdata;
 logic pending_completion,completion_error;
 async_fifo #(.WIDTH(49),.LGDEPTH(2)) requests(.wclk(clk),.wreset(reset),.wvalid(req_valid),.wready(req_ready),.wdata({req_write,req_addr,req_rows}),.rclk(mem_clk),.rreset(mem_reset),.rvalid(m_req_valid),.rready(m_req_ready),.rdata(rq));
 assign {m_req_write,m_req_addr,m_req_rows}=rq;
 async_fifo #(.WIDTH(128),.LGDEPTH(4)) writes(.wclk(clk),.wreset(reset),.wvalid(wr_valid),.wready(wr_ready),.wdata(wr_data),.rclk(mem_clk),.rreset(mem_reset),.rvalid(m_wr_valid),.rready(m_wr_ready),.rdata(m_wr_data));
 async_fifo #(.WIDTH(129),.LGDEPTH(4)) reads(.wclk(mem_clk),.wreset(mem_reset),.wvalid(m_rd_valid),.wready(m_rd_ready),.wdata({m_rd_last,m_rd_data}),.rclk(clk),.rreset(reset),.rvalid(rd_valid),.rready(rd_ready),.rdata(readq));
 assign {rd_last,rd_data}=readq;
 // Hold completion until its FIFO accepts it. DMA produces single-cycle pulses.
 always_ff @(posedge mem_clk)begin
 if(mem_reset)begin pending_completion<=0;completion_error<=0;end
 else begin
 if(pending_completion&&cr)pending_completion<=0;
 if(m_done)begin pending_completion<=1;completion_error<=m_error;end
 end end
 async_fifo #(.WIDTH(1),.LGDEPTH(2)) completions(.wclk(mem_clk),.wreset(mem_reset),.wvalid(pending_completion),.wready(cr),.wdata(completion_error),.rclk(clk),.rreset(reset),.rvalid(cv),.rready(!rd_valid),.rdata(cdata));
 assign dma_done=cv&&!rd_valid;assign dma_error=cv&&!rd_valid&&cdata;
endmodule
