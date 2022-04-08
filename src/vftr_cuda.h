#ifndef VFTR_CUDA_H
#define VFTR_CUDA_H

#include <cuda_runtime_api.h>

extern int vftr_n_cuda_devices;
extern struct cudaDeviceProp vftr_cuda_properties;

enum {T_CUDA_COMP, T_CUDA_MEMCP};

void vftr_cuda_info();
void vftr_setup_cuda();
void vftr_final_cuda();

typedef struct cuda_event_list_st {
  //const char *func_name;
  char *func_name;
  int n_calls;
  uint64_t memcpy_bytes;
  float t_acc[2];
  cudaEvent_t start;
  cudaEvent_t stop;
  struct cuda_event_list_st *next;
} cuda_event_list_t;

void vftr_cuda_flush_events (cuda_event_list_t **);

#endif
