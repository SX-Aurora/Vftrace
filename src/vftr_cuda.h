#ifndef VFTR_CUDA_H
#define VFTR_CUDA_H

#include <cuda_runtime.h>

void setup_vftr_cuda();
void final_vftr_cuda();

typedef struct cuda_event_list_st {
  //const char *func_name;
  char *func_name;
  int n_calls;
  uint64_t memcpy_bytes;
  float t_acc_compute;
  float t_acc_memcpy;
  cudaEvent_t start;
  cudaEvent_t stop;
  struct cuda_event_list_st *next;
} cuda_event_list_t;

void vftr_cuda_flush_events (cuda_event_list_t **);

#endif
