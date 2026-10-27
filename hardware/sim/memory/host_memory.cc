#include "mm.h"
#include <map>
#include <vector>
extern std::vector<std::map<long long,backing_data_t>> backing_mem_data;
extern "C" int ternscope_host_put(unsigned addr,unsigned byte) {if(backing_mem_data.empty()||backing_mem_data[0].empty())return -1;auto &b=backing_mem_data[0].begin()->second;if(addr>=b.size)return -2;b.data[addr]=byte;return 0;}
extern "C" int ternscope_host_get(unsigned addr) {if(backing_mem_data.empty()||backing_mem_data[0].empty())return -1;auto &b=backing_mem_data[0].begin()->second;if(addr>=b.size)return -2;return b.data[addr];}
