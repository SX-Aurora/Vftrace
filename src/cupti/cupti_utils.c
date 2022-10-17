#include <stdio.h>
#include <stdbool.h>

#include <cuda_runtime_api.h>

#include "vftrace_state.h"
#include "cupti_vftr_callbacks.h"

void vftr_show_used_gpu_info (FILE *fp) {
   struct cudaDeviceProp prop;
   int n_gpus = vftrace.cupti_state.n_devices;
   char *gpu_names[n_gpus];
   
   for (int i = 0; i < n_gpus; i++) {
      cudaGetDeviceProperties (&prop, i);
      gpu_names[i] = prop.name;
   }

   bool all_gpu_same = true;
   for (int i = 0; i < n_gpus; i++) {
      if (strcmp(gpu_names[i], prop.name)) {
	  all_gpu_same = false;
          break;
      }
   }

   fprintf (fp, "\n");
   
   if (n_gpus == 0) {
      fprintf (fp, "No GPUs available\n");
   } else if (all_gpu_same) {
      fprintf (fp, "Using %d GPUs: %s\n", n_gpus, prop.name);
   } else {
      fprintf (fp, "Using %d GPUs: \n", n_gpus);
      for (int i = 0; i < n_gpus; i++) {
         fprintf (fp, "   %d: %s\n", i, gpu_names[i]);
      }
   }
   
   if (n_gpus > 0) {
      char *visible_devices = getenv("CUDA_VISIBLE_DEVICES");
      fprintf (fp, "Visible GPUs: %s\n", visible_devices == NULL ? "all" : visible_devices);
   }
}

bool vftr_cupti_cbid_belongs_to_class (int cbid, int cbid_class) {
   bool is_compute =  cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020
                   || cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000;
   bool is_memcpy = cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020
                 || cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020;
   bool is_other = !is_compute && !is_memcpy;

   switch (cbid_class) {
      case T_CUPTI_COMP:
         return is_compute;
      case T_CUPTI_MEMCP:
         return is_memcpy;
      case T_CUPTI_OTHER:
         return is_other;
      default:
         return false; 
   }
}
