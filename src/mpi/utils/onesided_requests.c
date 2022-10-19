#include <mpi.h>

#include <stdlib.h>

#include "self_profile.h"
#include "vftrace_state.h"
#include "stack_types.h"
#include "thread_types.h"
#include "threadstack_types.h"
#include "profiling_types.h"
#include "rank_translate.h"
#include "requests.h"
#include "timer.h"
#include "mpi_logging.h"
#include "profiling.h"
#include "mpiprofiling.h"

void vftr_register_onesided_request(message_direction dir, int count,
                                    MPI_Datatype type, int peer_rank,
                                    MPI_Comm comm, MPI_Request request,
                                    long long tstart) {
   SELF_PROFILE_START_FUNCTION;
   // immediately return if peer is MPI_PROC_NULL as this is a dummy rank
   // with no effect on communication at all
   if (peer_rank == MPI_PROC_NULL) {
      SELF_PROFILE_END_FUNCTION;
      return;
   }

   vftr_request_t *new_request = vftr_register_request(dir, 1, &count, &type, -1, comm, request, 0, NULL, tstart);
   new_request->rank[0] = vftr_local2global_rank(comm, peer_rank);
   new_request->request_kind = onesided;
   new_request->persistent = false;
   SELF_PROFILE_END_FUNCTION;
}

void vftr_clear_completed_onesided_request(vftr_request_t *request) {
   // Test if the current request is still ongoing or completed
   // Note: MPI_Test is a destructive test. It will destroy the request
   //       Thus, running with and without vftr can lead to different program executions
   //       MPI_Request_get_status is a non destructive status check
   MPI_Status tmpStatus;
   int flag;
   PMPI_Request_get_status(request->request,
                           &flag,
                           &tmpStatus);

   // if the requested communication is finished write the communication out
   // and remove the request from the request list
   if (flag) {
      // record the time when the communication is finished
      // (might be to late, but thats as accurate as we can make it
      //  without violating the MPI-Standard)
      // Therefore: measures asynchronous communication with vftrace always
      //            yields to small bandwidth.
      long long tend = vftr_get_runtime_nsec ();

      // Every rank should already be translated to the global rank
      // by the register routine
      // Get the thread that called the function
      thread_t *my_thread = vftrace.process.threadtree.threads+request->callingthreadID;
      stack_t *my_stack = vftrace.process.stacktree.stacks+request->callingstackID;
      profile_t *my_profile = vftr_get_my_profile(my_stack, my_thread);

      // accumulate information for later use in the log file statistics
      vftr_accumulate_message_info(&(my_profile->mpiprof),
                                   vftrace.mpi_state,
                                   request->dir,
                                   request->count[0],
                                   request->type_idx[0],
                                   request->type_size[0],
                                   request->rank[0],
                                   request->tag,
                                   request->tstart, tend);
      // write the completed communication info to the vfd-file
      if (vftrace.environment.do_sampling.value.bool_val) {
         vftr_write_message_info(request->dir,
                                 request->count[0],
                                 request->type_idx[0],
                                 request->type_size[0],
                                 request->rank[0],
                                 request->tag,
                                 request->tstart,
                                 tend,
                                 request->callingstackID,
                                 request->callingthreadID);
      }

      // Take the request out of the list
      vftr_remove_request(request);
   }
}
