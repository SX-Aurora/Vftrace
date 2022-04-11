#ifndef VFTR_CUDA_H
#define VFTR_CUDA_H

#include <stdint.h>

extern int vftr_n_cuda_devices;

bool vftr_profile_cuda();

typedef struct cuda_event_list_st {
  char *func_name;
  int n_calls;
  uint64_t memcpy_bytes;
  float t_acc[2];
  struct cuda_event_list_st *next;
} cuda_event_list_t;

enum {T_CUDA_COMP, T_CUDA_MEMCP};

void vftr_cuda_flush_events (cuda_event_list_t **);
void vftr_cuda_info();
void vftr_setup_cuda();
void vftr_final_cuda();

#ifdef _CUPTI_AVAIL
#include <cuda_runtime_api.h>

extern struct cudaDeviceProp vftr_cuda_properties;

typedef struct cuda_event_list_internal_st {
  //const char *func_name;
  char *func_name;
  int n_calls;
  uint64_t memcpy_bytes;
  float t_acc[2];
  cudaEvent_t start;
  cudaEvent_t stop;
  struct cuda_event_list_internal_st *next;
} cuda_event_list_internal_t;

#endif

#endif
