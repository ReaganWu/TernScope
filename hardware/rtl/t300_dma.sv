`timescale 1ns/1ps
// 128b row client; single outstanding AXI64, <=64B bursts.
module t300_dma(input logic clk,reset,
 input logic req_valid,output logic req_ready,input logic req_write,
 input logic [31:0] req_addr,input logic [15:0] req_rows,
 input logic wr_valid,output logic wr_ready,input logic [127:0] wr_data,
 output logic rd_valid,input logic rd_ready,output logic [127:0] rd_data,output logic rd_last,
 output logic done,error,
 output logic aw_valid,input logic aw_ready,output logic [31:0] aw_addr,output logic [7:0] aw_len,
 output logic w_valid,input logic w_ready,output logic [63:0] w_data,output logic w_last,
 input logic b_valid,output logic b_ready,input logic [1:0] b_resp,
 output logic ar_valid,input logic ar_ready,output logic [31:0] ar_addr,output logic [7:0] ar_len,
 input logic r_valid,output logic r_ready,input logic [63:0] r_data,input logic r_last,input logic [1:0] r_resp);
 typedef enum logic [2:0] {IDLE,ADDR,WSTREAM,BRESP,RSTREAM,RDELIVER} st_t;
 st_t state; logic writing; logic [31:0] addr; logic [15:0] remain;
 logic [2:0] burst_rows; logic [3:0] beat; logic [127:0] bufdata; logic have_row;
 logic [15:0] sent_rows;
 assign req_ready=state==IDLE;
 assign aw_valid=state==ADDR&&writing; assign ar_valid=state==ADDR&&!writing;
 assign aw_addr=addr; assign ar_addr=addr;
 assign aw_len={4'b0,burst_rows,1'b0}-1; assign ar_len=aw_len;
 assign wr_ready=state==WSTREAM&&!have_row;
 assign w_valid=state==WSTREAM&&have_row;
 assign w_data=beat[0]?bufdata[127:64]:bufdata[63:0];
 assign w_last=beat==({1'b0,burst_rows}*2-1);
 assign b_ready=state==BRESP;
 assign r_ready=state==RSTREAM;
 assign rd_valid=state==RDELIVER; assign rd_data=bufdata;
 assign rd_last=remain==1;
 function automatic [2:0] chunk(input [31:0] a,input [15:0] n);
 logic [2:0] limit;
 begin limit=4-{1'b0,a[5:4]}; chunk=(n<limit)?n[2:0]:limit; end endfunction
 always_ff @(posedge clk) begin
 done<=0;
 if(reset) begin state<=IDLE; error<=0; have_row<=0; beat<=0; end
 else case(state)
 IDLE: if(req_valid) begin
 error<=0; writing<=req_write; addr<=req_addr; remain<=req_rows; beat<=0;have_row<=0;
 if(req_rows==0||req_addr[3:0]!=0) begin error<=1;done<=1;end
 else begin burst_rows<=chunk(req_addr,req_rows);state<=ADDR;end
 end
 ADDR: if((writing&&aw_ready)||(!writing&&ar_ready)) begin beat<=0;state<=writing?WSTREAM:RSTREAM;end
 WSTREAM: begin
 if(wr_valid&&wr_ready) begin bufdata<=wr_data;have_row<=1;end
 if(w_valid&&w_ready) begin
 beat<=beat+1;
 if(beat[0]) begin have_row<=0; remain<=remain-1;end
 if(w_last) state<=BRESP;
 end
 end
 BRESP: if(b_valid) begin
 if(b_resp!=0) error<=1;
 if(remain==0) begin state<=IDLE;done<=1;end
 else begin addr<=addr+{25'b0,burst_rows,4'b0};burst_rows<=chunk(addr+{25'b0,burst_rows,4'b0},remain);state<=ADDR;end
 end
 RSTREAM: if(r_valid) begin
 if(r_resp!=0) error<=1;
 if(r_last!=(beat==({1'b0,burst_rows}*2-1))) error<=1;
 if(!beat[0]) bufdata[63:0]<=r_data;
 else begin bufdata[127:64]<=r_data;state<=RDELIVER;end
 beat<=beat+1;
 end
 RDELIVER: if(rd_ready) begin
 remain<=remain-1;
 if(remain==1) begin state<=IDLE;done<=1;end
 else if(beat==({1'b0,burst_rows}*2)) begin
 addr<=addr+{25'b0,burst_rows,4'b0};burst_rows<=chunk(addr+{25'b0,burst_rows,4'b0},remain-1);state<=ADDR;
 end else state<=RSTREAM;
 end
 default:state<=IDLE;
 endcase
 end
endmodule
