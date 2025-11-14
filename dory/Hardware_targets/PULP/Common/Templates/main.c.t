
<%
l3_supported = DORY_HW_graph[0].HW_description['memory']['levels'] > 2
n_inputs = DORY_HW_graph[0].n_test_inputs
single_input = n_inputs == 1
%>\
% if not l3_supported:
#include "${prefix}input.h"
% else:
#include "mem.h"
% endif
#include "${prefix}network.h"

#include "pmsis.h"

% if verbose:
#define VERBOSE 1
% endif

% if sdk == 'pulp-sdk':
unsigned int PMU_set_voltage(unsigned int Voltage, unsigned int CheckFrequencies) {
  return 0;
}
% endif


void application(void * arg) 
{
  /*
      Opening of Filesystem and Ram
  */
% if l3_supported:
  mem_init();
  ${prefix}network_initialize();
% endif
  /*
    Allocating space for input
  */
  void *l2_buffer = pi_l2_malloc(${l2_buffer_size});
  if (NULL == l2_buffer) 
  {
#ifdef VERBOSE
    printf("ERROR: L2 buffer allocation failed.");
#endif
    pmsis_exit(-1);
  }
#ifdef VERBOSE
  printf("\nL2 Buffer alloc initial\t@ 0x%08x:\tOk\n", (unsigned int)l2_buffer);
#endif
  size_t l2_input_size = ${int(DORY_HW_graph[0].tiling_dimensions["L2"]["input_activation_memory"])};
  size_t input_size = 1000000;
  int initial_dir = 1;
% if l3_supported:
  void *ram_input = ram_malloc(input_size);
% endif
% if not single_input:
## multiple input
  for (int exec = 0; exec < ${n_inputs}; exec++) 
  {
% if l3_supported:
    load_file_to_ram(ram_input, ${prefix}Input_names[exec]);
    ram_read(l2_buffer, ram_input, l2_input_size);
% endif
    // run the inference
    ${prefix}network_run(l2_buffer, ${l2_buffer_size}, l2_buffer, ${"0" if single_input else "exec"}, initial_dir${f", {prefix}L2_input_h{' + exec * l2_input_size' if not single_input else ''}" if not l3_supported else ""});
  }
% else:
## single input
% if l3_supported:
  load_file_to_ram(ram_input, "${prefix}inputs.hex");
  ram_read(l2_buffer, ram_input, l2_input_size);
% endif

  // run the inference
  ${prefix}network_run(l2_buffer, ${l2_buffer_size}, l2_buffer, ${"0" if single_input else "exec"}, initial_dir${f", {prefix}L2_input_h{' + exec * l2_input_size' if not single_input else ''}" if not l3_supported else ""});
% endif

% if l3_supported:
  ram_free(ram_input, input_size);
  ${prefix}network_terminate();
% endif
  pi_l2_free(l2_buffer, ${l2_buffer_size});
}

int main () {
#ifndef TARGET_CHIP_FAMILY_GAP9
  PMU_set_voltage(1000, 0);
#else
  pi_pmu_voltage_set(PI_PMU_VOLTAGE_DOMAIN_CHIP, PI_PMU_VOLT_800);
#endif
  pi_time_wait_us(10000);
  pi_freq_set(PI_FREQ_DOMAIN_FC, ${fc_frequency});
  pi_time_wait_us(10000);
  pi_freq_set(PI_FREQ_DOMAIN_CL, ${cl_frequency});
  pi_time_wait_us(10000);
% if periph_frequency is not None:
  pi_freq_set(PI_FREQ_DOMAIN_PERIPH, ${periph_frequency});
  pi_time_wait_us(10000);
% endif

% if sdk == 'pulp-sdk':
  #if __PLATFORM__ == ARCHI_PLATFORM_FPGA
    *(int*)(ICACHE_PREFETCH) = 0xFFFF;
  #endif
% endif

  pmsis_kickoff((void*)application);
  return 0;
}
