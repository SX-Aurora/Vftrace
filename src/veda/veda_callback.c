#include <stdio.h>

#include <veda.h>

//#include "veda_callback_mem_alloc.h"
//#include "veda_callback_mem_free.h"
//#include "veda_callback_mem_cpy_htod.h"
//#include "veda_callback_mem_cpy_dtoh.h"
#include "veda_callback_lauch_kernel.h"
//#include "veda_callback_lauch_host.h"
//#include "veda_callback_hmem_cpy.h"
//#include "veda_callback_hmem_alloc.h"
//#include "veda_callback_hmem_free.h"

void vftr_veda_callback(VEDAprofiler_data* data, const int enter) {
   if (enter) {
      switch (data->type) {
         case VEDA_PROFILER_MEM_ALLOC:
            //vftr_veda_callback_mem_alloc_enter(data);
            break;
         case VEDA_PROFILER_MEM_FREE:
            //vftr_veda_callback_mem_free_enter(data);
            break;
         case VEDA_PROFILER_MEM_CPY_HTOD:
            //vftr_veda_callback_mem_cpy_htod_enter(data);
            break;
         case VEDA_PROFILER_MEM_CPY_DTOH:
            //vftr_veda_callback_mem_cpy_dtoh_enter(data);
            break;
         case VEDA_PROFILER_LAUNCH_KERNEL:
            vftr_veda_callback_lauch_kernel_enter(data);
            break;
         case VEDA_PROFILER_LAUNCH_HOST:
            //vftr_veda_callback_lauch_host_enter(data);
            break;
         case VEDA_PROFILER_HMEM_CPY:
            //vftr_veda_callback_hmem_cpy_enter(data);
            break;
         case VEDA_PROFILER_HMEM_ALLOC:
            //vftr_veda_callback_hmem_alloc_enter(data);
            break;
         case VEDA_PROFILER_HMEM_FREE:
            //vftr_veda_callback_hmem_free_enter(data);
            break;
         default:
            fprintf(stderr, "Encountered unknown veda-callback enter event %d\n",
                    data->type);
      }
   } else {
      switch (data->type) {
         case VEDA_PROFILER_MEM_ALLOC:
            //vftr_veda_callback_mem_alloc_exit(data);
            break;
         case VEDA_PROFILER_MEM_FREE:
            //vftr_veda_callback_mem_free_exit(data);
            break;
         case VEDA_PROFILER_MEM_CPY_HTOD:
            //vftr_veda_callback_mem_cpy_htod_exit(data);
            break;
         case VEDA_PROFILER_MEM_CPY_DTOH:
            //vftr_veda_callback_mem_cpy_dtoh_exit(data);
            break;
         case VEDA_PROFILER_LAUNCH_KERNEL:
            //vftr_veda_callback_lauch_kernel_exit(data);
            break;
         case VEDA_PROFILER_LAUNCH_HOST:
            //vftr_veda_callback_lauch_host_exit(data);
            break;
         case VEDA_PROFILER_HMEM_CPY:
            //vftr_veda_callback_hmem_cpy_exit(data);
            break;
         case VEDA_PROFILER_HMEM_ALLOC:
            //vftr_veda_callback_hmem_alloc_exit(data);
            break;
         case VEDA_PROFILER_HMEM_FREE:
            //vftr_veda_callback_hmem_free_exit(data);
            break;
         default:
            fprintf(stderr, "Encountered unknown veda-callback exit event %d\n",
                    data->type);
      }
   }
}
