module t300_ddr_harness #(parameter CLOCK_HZ=100000000)(input logic clk,reset,
 input logic req_valid,output logic req_ready,input logic req_write,input logic [31:0] req_addr,input logic [15:0] req_rows,
 input logic wr_valid,output logic wr_ready,input logic [127:0] wr_data,
 output logic rd_valid,input logic rd_ready,output logic [127:0] rd_data,output logic rd_last,output logic done,error);
 logic aw_valid,aw_ready,w_valid,w_ready,w_last,b_valid,b_ready,ar_valid,ar_ready,r_valid,r_ready,r_last;
 logic [31:0] aw_addr,ar_addr;logic [7:0] aw_len,ar_len;logic [63:0] w_data,r_data;logic [1:0] b_resp,r_resp;
 t300_dma dma(.*);
 SimDRAM #(.ADDR_BITS(32),.DATA_BITS(64),.ID_BITS(4),.MEM_SIZE(268435456),.LINE_SIZE(64),.CLOCK_HZ(CLOCK_HZ),.MEM_BASE(0),.CHIP_ID(0)) dram(
 .clock(clk),.reset(reset),
 .axi_aw_ready(aw_ready),.axi_aw_valid(aw_valid),.axi_aw_bits_addr(aw_addr),.axi_aw_bits_len(aw_len),
 .axi_aw_bits_size(3'd3),
 .axi_aw_bits_burst(2'd1),
 .axi_aw_bits_lock(1'b0),
 .axi_aw_bits_cache(4'b0),
 .axi_aw_bits_prot(3'b0),
 .axi_aw_bits_qos(4'b0),
 .axi_aw_bits_id(4'b0),
 .axi_ar_ready(ar_ready),.axi_ar_valid(ar_valid),.axi_ar_bits_addr(ar_addr),.axi_ar_bits_len(ar_len),
 .axi_ar_bits_size(3'd3),
 .axi_ar_bits_burst(2'd1),
 .axi_ar_bits_lock(1'b0),
 .axi_ar_bits_cache(4'b0),
 .axi_ar_bits_prot(3'b0),
 .axi_ar_bits_qos(4'b0),
 .axi_ar_bits_id(4'b0),
 .axi_w_ready(w_ready),.axi_w_valid(w_valid),.axi_w_bits_data(w_data),.axi_w_bits_last(w_last),.axi_w_bits_strb(8'hff),
 .axi_b_ready(b_ready),.axi_b_valid(b_valid),.axi_b_bits_resp(b_resp),.axi_b_bits_id(),
 .axi_r_ready(r_ready),.axi_r_valid(r_valid),.axi_r_bits_resp(r_resp),.axi_r_bits_data(r_data),.axi_r_bits_last(r_last),.axi_r_bits_id());
endmodule
