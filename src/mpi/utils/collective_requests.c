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

void vftr_register_collective_request(message_direction dir, int nmsg, int *count,
                                      MPI_Datatype *type, int *peer_rank,
                                      MPI_Comm comm, MPI_Request request,
                                      int n_tmp_ptr, void **tmp_ptrs,
                                      long long tstart) {
   SELF_PROFILE_START_FUNCTION;
   vftr_request_t *new_request = vftr_register_request(dir, nmsg, count, type, -1, comm, request, n_tmp_ptr, tmp_ptrs, tstart);
   int isintercom;
   PMPI_Comm_test_inter(comm, &isintercom);
   // translate the ranks to MPI_COMM_WORLD ranks
   if (isintercom) {
      for (int i=0; i<nmsg; i++) {
         new_request->rank[i] = vftr_remote2global_rank(comm, peer_rank[i]);
      }
   } else {
      for (int i=0; i<nmsg; i++) {
         // If a rank is -1 keep it. It is an invalid rank due to
         // non-periodic cartesian communicators.
         if (peer_rank[i] == -1) {
            new_request->rank[i] = -1;
         } else {
            new_request->rank[i] = vftr_local2global_rank(comm, peer_rank[i]);
         }
      }
   }
   new_request->request_kind = collective;
   new_request->persistent = false;
   SELF_PROFILE_END_FUNCTION;
}

void vftr_clear_completed_collective_request(vftr_request_t *request) {
   // Test if the current request is still ongoing or completed
   // Note: MPI_Test is a destructive test. It will destroy the request
   //       Thus, running with and without vftr can lead to different program executions
   //       MPI_Request_get_status is a non destructive status check
   int flag;
   PMPI_Request_get_status(request->request,
                           &flag,
                           MPI_STATUS_IGNORE);

   // if the requested communication is finished write the communication out
   // and remove the request from the request list
   if (flag) {
      // record the time when the communication is finished
      // (might be to late, but thats as accurate as we can make it
      //  without violating the MPI-Standard)
      // Therefore: measures asynchronous communication with vftrace always
      //            yields to small bandwidth.
      long long tend = vftr_get_runtime_nsec ();

      // Get the thread that called the function
      thread_t *my_thread = vftrace.process.threadtree.threads+request->callingthreadID;
      vftr_stack_t *my_stack = vftrace.process.stacktree.stacks+request->callingstackID;
      profile_t *my_profile = vftr_get_my_profile(my_stack, my_thread);

      // Every rank should already be translated to the global rank
      // by the register routine
      for (int i=0; i<request->nmsg; i++) {
         // if a rank is -1 skip the registering, as
         // it is an invalid rank due to non periodic
         // cartesian communicators.
         if (request->rank[i] != -1) {
            bool should_log_message = vftr_should_log_message_info(vftrace.mpi_state,
                                                                   request->rank[i]);
            if (should_log_message) {
               vftr_accumulate_message_info(&(my_profile->mpiprof),
                                            request->dir,
                                            request->count[i],
                                            request->type_idx[i],
                                            request->type_size[i],
                                            request->rank[i],
                                            request->tag,
                                            request->tstart, tend);
               if (vftrace.config.sampling.active.value) {
                  vftr_write_message_info(request->dir,
                                          request->count[i],
                                          request->type_idx[i],
                                          request->type_size[i],
                                          request->rank[i],
                                          request->tag,
                                          request->tstart,
                                          tend,
                                          request->callingstackID,
                                          request->callingthreadID);
               }
            }
         }
      }

      // Take the request out of the list
      vftr_remove_request(request);
   }
}
