#include "Vternscope_system.h"
#include "Vternscope_system___024root.h"
#include "verilated.h"
#include <array>
#include <vector>
#include <fstream>
#include <string>
#include <iostream>
#include <stdexcept>
extern "C" int ternscope_actual_dramsim_backend();
extern "C" uint64_t ternscope_ddr_audit(int);
using Row=std::array<uint32_t,4>;
extern "C" int ternscope_host_put(unsigned,unsigned);
extern "C" int ternscope_host_get(unsigned);
static std::vector<Row> readhex(const std::string&p){std::ifstream f(p);if(!f)throw std::runtime_error("missing "+p);std::vector<Row>v;std::string s;while(f>>s){if(s.size()!=32)throw std::runtime_error("bad hex "+p);Row a;for(int i=0;i<4;i++)a[i]=std::stoul(s.substr(24-i*8,8),nullptr,16);v.push_back(a);}return v;}
static bool match(const VlWide<4>&a,const Row&b){for(int i=0;i<4;i++)if(a[i]!=b[i])return false;return true;}
int main(int argc,char**argv){try{
 Verilated::commandArgs(argc,argv);std::string data="../reference/data";int frames=1;for(int i=1;i<argc;i++){std::string a=argv[i];if(a.rfind("+data=",0)==0)data=a.substr(6);if(a.rfind("+frames=",0)==0)frames=std::stoi(a.substr(8));}
 Vternscope_system d;uint64_t global=0;
 auto pre=[&](){d.clk=0;for(int h=0;h<5;h++){d.mem_clk=h%2;d.eval();}};
 auto post=[&](){d.clk=1;for(int h=5;h<10;h++){d.mem_clk=h%2;d.eval();}global++;};
 auto tick=[&](){pre();post();};
 d.reset=1;d.start=0;d.input_valid=0;d.host_ready=0;for(int i=0;i<8;i++)tick();d.reset=0;for(int i=0;i<4;i++)tick();
 if(!ternscope_actual_dramsim_backend())throw std::runtime_error("actual DRAMSim2 backend required; missing +dramsim");std::cout<<"BACKEND DRAMSim2 verified"<<std::endl;
 auto weights=readhex(data+"/weights.hex128");for(size_t n=0;n<weights.size();n++)for(int b=0;b<16;b++)if(ternscope_host_put(1048576+n*16+b,(weights[n][b/4]>>(8*(b%4)))&255))throw std::runtime_error("W2 preload failed");
 for(int iteration=0;iteration<frames;iteration++){
 int frame=iteration%4;auto input=readhex(data+"/inputs_frame"+std::to_string(frame)+".hex128");auto final=readhex(data+"/expected_frame"+std::to_string(frame)+"_final.hex128");
 std::array<std::vector<Row>,29>expected;std::array<std::vector<bool>,29>seen;for(int n=1;n<=28;n++){expected[n]=readhex(data+"/expected_frame"+std::to_string(frame)+"_n"+std::to_string(n)+".hex128");seen[n].resize(expected[n].size());}
 std::array<uint64_t,32>states{};size_t ins=0,outs=0,commits=0;uint64_t begin=global;uint64_t rb=d.read_bursts,wb=d.write_bursts,rbeat=d.read_beats,wbeat=d.write_beats,rwait=d.read_wait_cycles,wwait=d.write_wait_cycles;
 uint64_t audit_before[4];for(int k=0;k<4;k++)audit_before[k]=ternscope_ddr_audit(k);
 d.cold=iteration==0;d.start=1;tick();d.start=0;
 while(!d.done){
 if(global-begin>20000000)throw std::runtime_error("timeout frame "+std::to_string(iteration));
 d.input_valid=ins<input.size();if(ins<input.size())for(int k=0;k<4;k++)d.input_data[k]=input[ins][k];d.host_ready=1;pre();
 states[d.rootp->ternscope_system__DOT__core__DOT__state]++;bool in=d.input_valid&&d.input_ready, out=d.host_valid&&d.host_ready;
 if(out){if(outs>=final.size()||!match(d.host_data,final[outs]))throw std::runtime_error("host mismatch row "+std::to_string(outs));outs++;}
 post();if(in)ins++;
 if(d.debug_valid){int n=d.debug_node,r=d.debug_row;if(n<1||n>28||r>=int(expected[n].size()))throw std::runtime_error("debug index n="+std::to_string(n)+" row="+std::to_string(r));if(seen[n][r])throw std::runtime_error("duplicate debug row");if(!match(d.debug_data,expected[n][r]))throw std::runtime_error("node mismatch n="+std::to_string(n)+" row="+std::to_string(r));seen[n][r]=true;commits++;}
 if(d.node_done){int n=d.node_done_id;if(n<1||n>28)throw std::runtime_error("node_done id");for(bool b:seen[n])if(!b)throw std::runtime_error("node incomplete "+std::to_string(n));std::cout<<"NODE frame="<<iteration<<" n="<<n<<" elapsed="<<global-begin<<std::endl;}
 if(d.error)throw std::runtime_error("RTL error");
 }
 for(int n=1;n<=28;n++)for(bool b:seen[n])if(!b)throw std::runtime_error("missing node row");if(ins!=input.size()||outs!=final.size())throw std::runtime_error("IO incomplete");
 std::cout<<"PASS FRAME iteration="<<iteration<<" source="<<frame<<" cold="<<(iteration==0)<<" elapsed="<<global-begin<<" core_hz=100000000 mem_hz=500000000 core_cycles="<<d.cycles<<" ingress_rows="<<ins<<" host_rows="<<outs<<" node_commits="<<commits<<" read_bursts="<<d.read_bursts-rb<<" write_bursts="<<d.write_bursts-wb<<" read_bytes="<<(d.read_beats-rbeat)*8<<" write_bytes="<<(d.write_beats-wbeat)*8<<" read_wait_cycles="<<d.read_wait_cycles-rwait<<" write_wait_cycles="<<d.write_wait_cycles-wwait<<std::endl;std::cout<<"DDR_AUDIT frame="<<iteration<<" aw_valid_not_ready="<<ternscope_ddr_audit(0)-audit_before[0]<<" w_valid_not_ready="<<ternscope_ddr_audit(1)-audit_before[1]<<" b_ready_not_valid="<<ternscope_ddr_audit(2)-audit_before[2]<<" completed_write_bursts="<<ternscope_ddr_audit(3)-audit_before[3]<<std::endl;std::cout<<"STATES";for(int st=0;st<32;st++)std::cout<<" "<<st<<":"<<states[st];std::cout<<std::endl;tick();
 }
 d.final();return 0;
 }catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<std::endl;return 1;}}
