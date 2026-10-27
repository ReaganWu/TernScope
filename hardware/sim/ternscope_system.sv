`timescale 1ns/1ps
module ternscope_system #(parameter integer LUT_ENTRIES=5376)(input logic clk,mem_clk,reset,start,cold,
 input logic input_valid,output logic input_ready,input logic [127:0] input_data,
 output logic host_valid,input logic host_ready,output logic [127:0] host_data,
 output logic done,error,output logic [63:0] cycles,
 output logic debug_valid,output logic [5:0] debug_node,output logic [15:0] debug_row,output logic [127:0] debug_data,
 output logic node_done,output logic [5:0] node_done_id,
 output logic [63:0] read_bursts,write_bursts,read_beats,write_beats,read_wait_cycles,write_wait_cycles);
 logic req_valid,req_ready,req_write,wr_valid,wr_ready,rd_valid,rd_ready,rd_last,dma_done,dma_error;
 logic [31:0] req_addr;logic[15:0]req_rows;logic[127:0]wr_data,rd_data;
 ternscope_core #(.LUT_ENTRIES(LUT_ENTRIES)) core(.*);
 logic m_req_valid,m_req_ready,m_req_write,m_wr_valid,m_wr_ready,m_rd_valid,m_rd_ready,m_rd_last,m_done,m_error;
 logic[31:0]m_req_addr;logic[15:0]m_req_rows;logic[127:0]m_wr_data,m_rd_data;
 // Reset deassertion synchronized independently in each clock domain.
 (* ASYNC_REG="TRUE" *) logic[1:0]mr;
 always_ff @(posedge mem_clk or posedge reset) if(reset)mr<=2'b11;else mr<={mr[0],1'b0};
 ternscope_row_cdc crossing(.clk(clk),.reset(reset),.mem_clk(mem_clk),.mem_reset(mr[1]),.*);
 ternscope_ddr_harness #(.CLOCK_HZ(500000000)) memory(.clk(mem_clk),.reset(mr[1]),.req_valid(m_req_valid),.req_ready(m_req_ready),.req_write(m_req_write),.req_addr(m_req_addr),.req_rows(m_req_rows),.wr_valid(m_wr_valid),.wr_ready(m_wr_ready),.wr_data(m_wr_data),.rd_valid(m_rd_valid),.rd_ready(m_rd_ready),.rd_data(m_rd_data),.rd_last(m_rd_last),.done(m_done),.error(m_error));
always_ff @(posedge mem_clk) begin
 if(mr[1]) begin read_bursts<=0;write_bursts<=0;read_beats<=0;write_beats<=0;read_wait_cycles<=0;write_wait_cycles<=0;end
 else begin
 if(memory.ar_valid&&memory.ar_ready)read_bursts<=read_bursts+1;
 if(memory.aw_valid&&memory.aw_ready)write_bursts<=write_bursts+1;
 if(memory.r_valid&&memory.r_ready)read_beats<=read_beats+1;
 if(memory.w_valid&&memory.w_ready)write_beats<=write_beats+1;
 if(memory.r_ready&&!memory.r_valid)read_wait_cycles<=read_wait_cycles+1;
 if(memory.w_valid&&!memory.w_ready)write_wait_cycles<=write_wait_cycles+1;
 end
end
endmodule
