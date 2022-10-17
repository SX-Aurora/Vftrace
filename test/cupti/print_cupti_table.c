#include "environment_types.h"
#include "environment.h"
#include "stack_types.h"
#include "collated_stack_types.h"
#include "collate_stacks.h"
#include "collate_profiles.h"

#include "cupti_init_final.h"
#include "cupti_vftr_callbacks.h"
#include "cuptiprofiling_types.h"
#include "cupti_ranklogfile.h"
#include "cupti_logfile.h"

#include "dummy_stacktree.h"

// COMPUTE_CBIDS: CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020,
//                CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000
// MEMCPY_CBIDS:  CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020,
//                CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020

int main(int argc, char **argv) {

   environment_t environment;
   environment = vftr_read_environment();

   vftr_set_ngpus ();

   vftr_init_dummy_stacktree (10);
   vftr_register_dummy_call_stack ("func0<init", 1);
   vftr_register_dummy_call_stack ("cudafunc1<init", 2);
   vftr_register_dummy_cupti_stack ("cudafunc1<init", CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020,
                                    1000.0, CUPTI_NOCOPY, 0);
   vftr_register_dummy_cupti_stack ("cudafunc2<func0<init", CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020,
				    1000.0, CUPTI_COPY_IN, 1048576);
   vftr_register_dummy_cupti_stack ("cudafunc3<func0<init", CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020,
				    20000.0, CUPTI_COPY_OUT, 524288);
   vftr_register_dummy_call_stack ("cudafunc1<init", 2);
   for (int i = 0; i < 4; i++) {
      vftr_register_dummy_cupti_stack ("cudafunc1<init", CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020,
                                       1000.0, CUPTI_NOCOPY, 0);
   }
   for (int i = 0; i < 3; i++) {
      vftr_register_dummy_cupti_stack ("cudafunc4<func0<init", CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020,
				       10000.0, CUPTI_NOCOPY, 0);
   }


   stacktree_t stacktree = vftr_get_dummy_stacktree();
   
   collated_stacktree_t collated_stacktree = vftr_collate_stacks(&stacktree);
   vftr_collate_profiles(&collated_stacktree, &stacktree);

   fprintf (stdout, "Ranklogfile: \n");
   vftr_write_ranklogfile_cupti_table(stdout, stacktree, environment);
   fprintf (stdout, "Logfile: \n");
   vftr_write_logfile_cupti_table (stdout, collated_stacktree, environment);
}

