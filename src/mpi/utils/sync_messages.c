#include <mpi.h>

#include "self_profile.h"
#include "vftrace_state.h"
#include "mpi_logging.h"
#include "mpi_logging.h"
#include "mpi_util_types.h"
#include "rank_translate.h"
#include "stack_types.h"
#include "thread_types.h"
#include "threadstack_types.h"
#include "profiling_types.h"
#include "mpiprofiling_types.h"
#include "stacks.h"
#include "threads.h"
#include "threadstacks.h"
#include "profiling.h"
#include "mpiprofiling.h"

// store message info for synchronous mpi-communication
void vftr_store_sync_message_info(message_direction dir, int count, MPI_Datatype type,
                                  int peer_rank, int tag, MPI_Comm comm,
                                  long long tstart, long long tend) {
   SELF_PROFILE_START_FUNCTION;
   // only continue if sampling and mpi_loggin is enabled
   if (vftr_no_mpi_logging()) {
      SELF_PROFILE_END_FUNCTION;
      return;
   }

   // immediately return if peer is MPI_PROC_NULL as this is a dummy rank
   // with no effect on communication at all
   if (peer_rank == MPI_PROC_NULL) {
      SELF_PROFILE_END_FUNCTION;
      return;
   }

   // translate rank to global in case the communicator is not global
   int rank = peer_rank;
   if (comm != MPI_COMM_WORLD) {
     // check if the communicator is an intercommunicator
     int isintercom;
     PMPI_Comm_test_inter(comm, &isintercom);
     if (isintercom) {
        rank = vftr_remote2global_rank(comm, peer_rank);
     } else {
        rank = vftr_local2global_rank(comm, peer_rank);
     }
   }

   int type_idx = vftr_get_mpitype_idx(type);
   int type_size;
   if (type != MPI_DATATYPE_NULL) {
      PMPI_Type_size(type, &type_size);
   } else {
      type_size = 0;
   }

   // Get the thread that called the function
   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);
   stack_t *my_stack = vftrace.process.stacktree.stacks+my_threadstack->stackID;
   profile_t *my_profile = vftr_get_my_profile(my_stack, my_thread);

   // accumulate information for later use in the log file statistics
   vftr_accumulate_message_info(&(my_profile->mpiprof),
                                vftrace.mpi_state,
                                dir, count,
                                type_idx, type_size,
                                rank, tag, tstart, tend);

   // write message info to vfd-file
   if (vftrace.environment.do_sampling.value.bool_val) {
      vftr_write_message_info(dir, count, type_idx, type_size,
                              rank, tag, tstart, tend,
                              my_threadstack->stackID,
                              my_thread->threadID);
   }

   SELF_PROFILE_END_FUNCTION;
}
