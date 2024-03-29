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

void vftr_register_p2p_request(message_direction dir, int count,
                               MPI_Datatype type, int peer_rank, int tag,
                               MPI_Comm comm, MPI_Request request,
                               long long tstart) {
   SELF_PROFILE_START_FUNCTION;
   // immediately return if peer is MPI_PROC_NULL as this is a dummy rank
   // with no effect on communication at all
   if (peer_rank == MPI_PROC_NULL) {
      SELF_PROFILE_END_FUNCTION;
      return;
   }

   vftr_request_t *new_request = vftr_register_request(dir, 1, &count, &type, tag, comm, request, 0, NULL, tstart);
   new_request->rank[0] = peer_rank;
   new_request->request_kind = p2p;
   new_request->persistent = false;
   SELF_PROFILE_END_FUNCTION;
}

void vftr_register_pers_p2p_request(message_direction dir, int count,
                                    MPI_Datatype type, int peer_rank, int tag,
                                    MPI_Comm comm, MPI_Request request) {
   SELF_PROFILE_START_FUNCTION;
   // immediately return if peer is MPI_PROC_NULL as this is a dummy rank
   // with no effect on communication at all
   if (peer_rank == MPI_PROC_NULL) {
      SELF_PROFILE_END_FUNCTION;
      return;
   }

   vftr_request_t *new_request = vftr_register_request(dir, 1, &count, &type, tag, comm, request, 0, NULL, 0);
   new_request->rank[0] = peer_rank;
   new_request->request_kind = p2p;
   new_request->persistent = true;
   new_request->active = false;
   SELF_PROFILE_END_FUNCTION;
}

void vftr_clear_completed_p2p_request(vftr_request_t *request) {
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

      // extract rank and tag from the completed communication status
      // (if necessary) this is to avoid errors with wildcard usage
      if (request->tag == MPI_ANY_TAG) {
         request->tag = tmpStatus.MPI_TAG;
      }
      if (request->rank[0] == MPI_ANY_SOURCE) {
         request->rank[0] = tmpStatus.MPI_SOURCE;
         request->rank[0] = vftr_local2global_rank(request->comm,
                                                           request->rank[0]);
      }

      // Get the actual amount of transferred data if it is a receive operation
      // only if it was a point2point communication
      if (request->dir == recv) {
         int tmpcount;
         PMPI_Get_count(&tmpStatus, request->type[0], &tmpcount);
         if (tmpcount != MPI_UNDEFINED) {
            request->count[0] = tmpcount;
         }
      }
      // Get the thread that called the function
      thread_t *my_thread = vftrace.process.threadtree.threads+request->callingthreadID;
      vftr_stack_t *my_stack = vftrace.process.stacktree.stacks+request->callingstackID;
      profile_t *my_profile = vftr_get_my_profile(my_stack, my_thread);

      // accumulate information for later use in the log file statistics
      bool should_log_message = vftr_should_log_message_info(vftrace.mpi_state,
                                                             request->rank[0]);
      if (should_log_message) {
         vftr_accumulate_message_info(&(my_profile->mpiprof),
                                      request->dir,
                                      request->count[0],
                                      request->type_idx[0],
                                      request->type_size[0],
                                      request->rank[0],
                                      request->tag,
                                      request->tstart, tend);
         // write the completed communication info to the vfd-file
         if (vftrace.config.sampling.active.value) {
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
      }

      if (request->persistent) {
         request->active = false;
         if (request->marked_for_deallocation) {
            PMPI_Request_free(&(request->request));
         }
      } else {
         if (request->marked_for_deallocation) {
            PMPI_Request_free(&(request->request));
         }
         vftr_remove_request(request);
      }
   }
}
