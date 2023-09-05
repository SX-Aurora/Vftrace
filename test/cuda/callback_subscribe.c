#include <stdio.h>

#include "cupti_init_final.h"
#include "cudaprofiling_types.h"
#include "cudaprofiling.h"
#include "cupti_vftr_callbacks.h"

cudaprofile_t dummy_cudaprof[2];

void CUPTIAPI dummy_cupti_callback (void *userdata,
                                    CUpti_CallbackDomain domain,
                                    CUpti_CallbackId cbid,
                                    const CUpti_CallbackData *cb_info) {
   printf ("cbid: %d\n", cbid);
   cudaprofile_t *this_prof;
   if (cbid == 20) {
      this_prof = &(dummy_cudaprof[0]);
   } else if (cbid == 22) {
      this_prof = &(dummy_cudaprof[1]);
   }

   if (cb_info->callbackSite == CUPTI_API_ENTER) {
      vftr_accumulate_cudaprofiling (this_prof, cbid, 1, -1.0, CUDA_COPY_IN, 1000);
   } else if (cb_info->callbackSite == CUPTI_API_EXIT) {
      vftr_accumulate_cudaprofiling (this_prof, cbid, 0, +2.0, CUDA_COPY_OUT, 1000);
   }
}

int main (int argc, char *argv[]) {
   
   dummy_cudaprof[0] = vftr_new_cudaprofiling();
   dummy_cudaprof[1] = vftr_new_cudaprofiling();
   (void)vftr_init_cupti (dummy_cupti_callback);
   int *field;
   cudaMalloc((void**)&field, sizeof(int));   
   cudaFree(field);


   printf ("cbids: %d %d\n", dummy_cudaprof[0].cbid, dummy_cudaprof[1].cbid);
   printf ("calls: %d %d\n", dummy_cudaprof[0].n_calls[0] + dummy_cudaprof[0].n_calls[1],
                             dummy_cudaprof[1].n_calls[0] + dummy_cudaprof[1].n_calls[1]);
   printf ("t_ms: %.3f %.3f\n", dummy_cudaprof[0].t_ms, dummy_cudaprof[1].t_ms);
   printf ("memcpy in: %lld %lld\n", dummy_cudaprof[0].memcpy_bytes[CUDA_COPY_IN],
                                     dummy_cudaprof[1].memcpy_bytes[CUDA_COPY_IN]);
   printf ("memcpy out: %lld %lld\n", dummy_cudaprof[0].memcpy_bytes[CUDA_COPY_OUT],
                                      dummy_cudaprof[1].memcpy_bytes[CUDA_COPY_OUT]);
   return 0;
}
