#ifndef VFTR_CUDA_H
#define VFTR_CUDA_H

#ifdef __cplusplus
extern "C"
#endif
void setup_vftr_cuda();
#ifdef __cplusplus
extern "C"
#endif
void final_vftr_cuda();

typedef struct RuntimeApiTrace_st {
  const char *functionName;
  uint64_t startTimestamp;
  uint64_t endTimestamp;
  uint64_t t_acc_compute;
  uint64_t t_acc_memcpy;
  uint64_t n_calls;
  size_t memcpy_bytes;
  //enum cudaMemcpyKind memcpy_kind;
  struct RuntimeApiTrace_st *next;
} RuntimeApiTrace_t;

typedef struct kernel_list_st {
  const char *kernel_name;
  struct kernel_list_st *next;
} kernel_list_t;

#ifdef __cplusplus
extern "C"
#endif
void vftr_cuda_flush_trace (RuntimeApiTrace_t **);

#ifdef __cplusplus
extern "C"
#endif
void displayTimestamps (RuntimeApiTrace_t *);

#endif
