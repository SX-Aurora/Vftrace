#include <mpi.h>

#include "self_profile.h"
#include "thread_types.h"
#include "threads.h"
#include "threadstack_types.h"
#include "threadstacks.h"
#include "stack_types.h"
#include "stacks.h"
#include "profiling_types.h"
#include "profiling.h"
#include "mpiprofiling.h"
#include "timer.h"
#include "onesided_requests.h"

int vftr_MPI_Raccumulate(const void *origin_addr, int origin_count,
                         MPI_Datatype origin_datatype, int target_rank,
                         MPI_Aint target_disp, int target_count,
                         MPI_Datatype target_datatype, MPI_Op op,
                         MPI_Win win, MPI_Request *request) {
   long long tstart = vftr_get_runtime_nsec();
   int retVal = PMPI_Raccumulate(origin_addr, origin_count, origin_datatype,
                                 target_rank, target_disp, target_count,
                                 target_datatype, op, win, request);

   SELF_PROFILE_START_FUNCTION;
   long long t2start = vftr_get_runtime_nsec();
   // Need to figure out the partner rank in a known communicator to store info
   MPI_Group local_group;
   PMPI_Win_get_group(win, &local_group);

   MPI_Group global_group;
   PMPI_Comm_group(MPI_COMM_WORLD, &global_group);

   int global_rank;
   PMPI_Group_translate_ranks(local_group,
                              1,
                              &target_rank,
                              global_group,
                              &global_rank);

   vftr_register_onesided_request(send, origin_count, origin_datatype,
                                  global_rank, MPI_COMM_WORLD, *request, tstart);

   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);
   vftr_stack_t *my_stack = vftrace.process.stacktree.stacks+my_threadstack->stackID;
   profile_t *my_profile = vftr_get_my_profile(my_stack, my_thread);
   long long t2end = vftr_get_runtime_nsec();

   vftr_accumulate_mpiprofiling_overhead(&(my_profile->mpiprof), t2end-t2start);

   SELF_PROFILE_END_FUNCTION;
   return retVal;
}
