#include "net_utils.h"
#include "pmsis.h"

void print_perf(const char *name, const int cycles, const int macs) 
{
  float perf = (float) macs / cycles;
  printf("\n%s performance:\n", name);
  printf("  - num cycles: %d\n", cycles);
  printf("  - MACs: %d\n", macs );
  printf("  - MAC/cycle: %g\n", perf);
  printf("  - n. of Cores: %d\n\n", NUM_CORES);
}

void checksum(const char *name, const uint8_t *d, size_t size, uint32_t sum_true) 
{
  uint32_t sum = 0;
  for (int i = 0; i < size; i++) sum += d[i];

  printf("Checking %s: Checksum ", name);
  if (sum_true == sum)
    printf("OK\n");
  else if (sum_true == -1)
    printf("sum: %d\n", sum);
  else
    printf("Failed: true [%u] vs. calculated [%u]\n", sum_true, sum);
}

void print_layer_args(unsigned int *args)
{
  printf("===== Args dump =====\n");
  printf("l3_x        = 0x%08X\n", args[0]);
  printf("l3_y        = 0x%08X\n", args[1]);
  printf("l3_W        = 0x%08X\n", args[2]);
  printf("l2_x        = 0x%08X\n", args[3]);
  printf("l2_x_2      = 0x%08X\n", args[4]);
  printf("l2_y        = 0x%08X\n", args[5]);
  printf("l2_W        = 0x%08X\n", args[6]);
  printf("l1_buffer   = 0x%08X\n", args[7]);
  printf("hyperram    = 0x%08X\n", args[8]);
  printf("out_mult_in = %u\n", args[9]);
  printf("out_shift_in= %u\n", args[10]);
  printf("=====================\n");
}

void print_DMA_transfer(DMA_copy *copy, const char *label, unsigned L2_base_addr, unsigned L1_base_addr)
{ 
  char direction[10];
  if (copy->dir == 0)
    sprintf(direction, "L1 -> L2");
  else
    sprintf(direction, "L2 -> L1");

  unsigned int length = copy->length_1d_copy * copy->number_of_1d_copies * copy->number_of_2d_copies;
  printf("\n %s DMA transfer\t%u bytes dir: %s\n L2: 0x%08X + %u \n L1: 0x%08X + %u \n", 
    label, length, direction, L2_base_addr, ((unsigned)copy->ext - L2_base_addr), L1_base_addr, ((unsigned)copy->loc - L1_base_addr));
}

void debug_print_tensor(const DMA_copy *copy, const char *label, size_t max_elems)
{
  const uint8_t *ptr = NULL;
  if (copy-> dir)
    ptr = copy->loc;
  else
    ptr = copy->ext;

  size_t length = copy->length_1d_copy * copy->number_of_1d_copies * copy->number_of_2d_copies;

  printf("===== Dump of %s =====\n", label);
  printf("Address: %p | Length (bytes): %zu\n", ptr, length);
  size_t limit = (length < max_elems) ? length : max_elems;
  for (size_t i = 0; i < limit; i++)
  {
    printf("%02X ", ptr[i]);
    if ((i + 1) % 20 == 0)
      printf("\n");
  }
  if (limit % 20 != 0)
    printf("\n");

  if (length > max_elems)
    printf("... (showing first %zu of %zu bytes)\n", max_elems, length);
}
