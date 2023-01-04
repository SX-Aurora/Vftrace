#include <mpi.h>

#include "configuration_types.h"
#include "configuration.h"
#include "stack_types.h"
#include "collated_stack_types.h"
#include "collate_stacks.h"
#include "collate_profiles.h"

#include "cupti_init_final.h"
#include "cupti_vftr_callbacks.h"
#include "cudaprofiling_types.h"
#include "cuda_ranklogfile.h"
#include "cuda_logfile.h"

#include "dummy_stacktree.h"

// COMPUTE_CBIDS: CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020,
//                CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000
// MEMCPY_CBIDS:  CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020,
//                CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020

int main(int argc, char **argv) {
   PMPI_Init (&argc, &argv);
   int nranks, myrank;
   PMPI_Comm_size (MPI_COMM_WORLD, &nranks);
   if (nranks != 2) {
      fprintf (stderr, "This test requires exactly two ranks, "
               "but was started with %d\n", nranks);
      return 1;
   }
   PMPI_Comm_rank (MPI_COMM_WORLD, &myrank);

   config_t config;
   config = vftr_read_config();

   vftr_set_ngpus ();

   vftr_init_dummy_stacktree (10);
   if (myrank == 0) {
      vftr_register_dummy_call_stack ("func0<init", 1);
      vftr_register_dummy_call_stack ("cudafunc1<init", 2);
      vftr_register_dummy_cuda_stack ("cudafunc1<init", CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020,
                                       1000.0, CUDA_NOCOPY, 0);
      vftr_register_dummy_cuda_stack ("cudafunc2<func0<init", CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020,
           			    1000.0, CUDA_COPY_IN, 1048576);
      vftr_register_dummy_cuda_stack ("cudafunc3<func0<init", CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020,
           			    20000.0, CUDA_COPY_OUT, 524288);
      vftr_register_dummy_call_stack ("cudafunc1<init", 2);
      for (int i = 0; i < 3; i++) {
         vftr_register_dummy_cuda_stack ("cudafunc4<func0<init", CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020,
           			       10000.0, CUDA_NOCOPY, 0);
      }
   } else {
      vftr_register_dummy_call_stack ("func0<init", 1);
      vftr_register_dummy_call_stack ("cudafunc1<init", 2);
      vftr_register_dummy_cuda_stack ("cudafunc1<init", CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020,
                                       2000.0, CUDA_NOCOPY, 0);
      vftr_register_dummy_cuda_stack ("cudafunc2<func0<init", CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020,
           			    2000.0, CUDA_COPY_IN, 6758401);
      vftr_register_dummy_cuda_stack ("cudafunc3<func0<init", CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020,
           			    40000.0, CUDA_COPY_OUT, 882425);
      vftr_register_dummy_call_stack ("cudafunc1<init", 2);
      for (int i = 0; i < 5; i++) {
         vftr_register_dummy_cuda_stack ("cudafunc4<func0<init", CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020,
           			       20000.0, CUDA_NOCOPY, 0);
      }
      // Additional cudafunc not present on rank 0
      vftr_register_dummy_cuda_stack ("cudafunc5<func0<init", CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020,
                                      10000.0, CUDA_NOCOPY, 0);
   }

   stacktree_t stacktree = vftr_get_dummy_stacktree();
   collated_stacktree_t collated_stacktree = vftr_collate_stacks(&stacktree);
   vftr_collate_profiles (&collated_stacktree, &stacktree);

   for (int i = 0; i < nranks; i++) {
      if (myrank == i) {
        fprintf (stdout, "Ranklogfile for rank %d: \n", i);
        vftr_write_ranklogfile_cuda_table(stdout, stacktree, config);
      }
      PMPI_Barrier(MPI_COMM_WORLD);
   }
   if (myrank == 0) {
     fprintf (stdout, "Collated logfile: \n");
     vftr_write_logfile_cuda_table (stdout, collated_stacktree, config);
   }

   PMPI_Finalize();
}
