#ifndef VFTR_CUDA_H
#define VFTR_CUDA_H

void setup_vftr_cuda();
void final_vftr_cuda();

typedef struct cupti_trace_st {
  const char *func_name;
  uint64_t ts_start;
  uint64_t ts_end;
  uint64_t t_acc_compute;
  uint64_t t_acc_memcpy;
  uint64_t n_calls;
  size_t memcpy_bytes;
  struct cupti_trace_st *next;
} cupti_trace_t;

void vftr_cuda_flush_trace (cupti_trace_t **);

#endif
