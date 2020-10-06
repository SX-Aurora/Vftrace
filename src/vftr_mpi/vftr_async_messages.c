/*
   This file is part of Vftrace.

   Vftrace is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.

   Vftrace is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License along
   with this program; if not, write to the Free Software Foundation, Inc.,
   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/


#ifdef _MPI
#include <mpi.h>

#include <stdlib.h>

#include "vftr_mpi_utils.h"
#include "vftr_environment.h"
#include "vftr_filewrite.h"
#include "vftr_timer.h"
#include "vftr_pause.h"
#include "vftr_async_messages.h"

vftr_request_list_t *vftr_open_request_list = NULL;

// store message info for asynchronous mpi-communication
void vftr_register_request(vftr_direction dir, vftr_comm_t communication_type,
                           int nmsg, int *count, MPI_Datatype *type,
                           int *peer_rank, int tag, MPI_Comm comm,
                           MPI_Request request, long long tstart) {

   // only continue if sampling and mpi_loggin is enabled
   bool mpi_log = vftr_environment->mpi_log->value;
   if (vftr_off() || !mpi_log || vftr_paused) return;

   // immediately return if peer is MPI_PROC_NULL as this is a dummy rank
   // with no effect on communication at all
   if (peer_rank[0] == MPI_PROC_NULL) {return;}

   // create new request list entry
   vftr_request_list_t *new_open_request = (vftr_request_list_t*) 
      malloc(sizeof(vftr_request_list_t));
   new_open_request->request   = request;
   new_open_request->comm      = comm;
   new_open_request->communication_type = communication_type;
   new_open_request->nmsg      = nmsg;
   new_open_request->dir       = dir;
   new_open_request->count     = (int*) malloc(sizeof(int)*nmsg);
   for (int i=0; i<nmsg; i++) {
      new_open_request->count[i] = count[i];
   }
   new_open_request->type      = (MPI_Datatype*) malloc(sizeof(MPI_Datatype)*nmsg);
   for (int i=0; i<nmsg; i++) {
      new_open_request->type[i] = type[i];
   }
   new_open_request->type_idx  = (int*) malloc(sizeof(int)*nmsg);
   new_open_request->type_size = (int*) malloc(sizeof(int)*nmsg);
   for (int i=0; i<nmsg; i++) {
      // Determine type index in vftrace type list
      // and its size in bytes
      int type_idx = vftr_get_mpitype_idx(type[i]);
      int type_size;
      if (type[i] != MPI_DATATYPE_NULL) {
         PMPI_Type_size(type[i], &type_size);
      } else {
         type_size = 0;
      }
      new_open_request->type_idx[i]  = type_idx;
      new_open_request->type_size[i] = type_size;
   }

   // rank and tag are stored 
   // If they are wildcards (like MPI_ANY_SOURCE, MPI_ANY_TAG) they will be overwritten
   // as soon as the communication is complieted with the real values
   // extracted from the completed communication status 
   // This can only happen for point2point communication
   new_open_request->tag = tag;
   new_open_request->rank = (int*) malloc(sizeof(int)*nmsg);
   if (peer_rank[0] != MPI_ANY_SOURCE) {
      // check if the communicator is an intercommunicator
      int isintercom;
      PMPI_Comm_test_inter(comm, &isintercom);
      if (isintercom) {
         for (int i=0; i<nmsg; i++) {
            new_open_request->rank[i] = vftr_remote2global_rank(comm, peer_rank[i]);
         }
      } else {
         for (int i=0; i<nmsg; i++) {
            new_open_request->rank[i] = vftr_local2global_rank(comm, peer_rank[i]);
         }
      }
   } else {
      for (int i=0; i<nmsg; i++) {
         new_open_request->rank[i] = MPI_ANY_SOURCE;
      }
   }
   new_open_request->tstart    = tstart;

   // prepend message info to list for open communication
   if (vftr_open_request_list == NULL) {
      // list is empty. this is the first element
      vftr_open_request_list = new_open_request;
      vftr_open_request_list->prev = NULL;
      vftr_open_request_list->next = NULL;
   } else {
      // list contains entries
      new_open_request->next = vftr_open_request_list;
      new_open_request->prev = NULL;
      vftr_open_request_list->prev = new_open_request;
      vftr_open_request_list = new_open_request;
   }
}

//void vftr_register_P2P_request(vftr_direction dir, int count,
//                               MPI_Datatype type, int peer_rank, int tag,
//                               MPI_Comm comm, MPI_Request request,
//                               long long tstart) {
//   vftr_register_request(dir, vftr_comm_P2P, 1, &count, &type,
//                         &peer_rank, tag, comm, request, tstart);
//}

//void vftr_register_onesided_request(vftr_direction dir, int count,
//                                    MPI_Datatype type, int peer_rank,
//                                    MPI_Comm comm, MPI_Request request,
//                                    long long tstart) {
//   vftr_register_request(dir, vftr_comm_onesided, 1, &count, &type,
//                         &peer_rank, -1, comm, request, tstart);
//}

void vftr_register_collective_request(vftr_direction dir, int nmsg, int *count,
                                      MPI_Datatype *type, int *peer_rank,
                                      MPI_Comm comm, MPI_Request request,
                                      long long tstart) {
   vftr_register_request(dir, vftr_comm_collective, nmsg, count, type, peer_rank,
                         -1, comm, request, tstart);
}
                           
// test the entire list of open request for completed communication
void vftr_clear_completed_request() {
   // only continue if sampling and mpi_loggin is enabled
   bool mpi_log = vftr_environment->mpi_log->value;
   if (vftr_off() || !mpi_log) return;

   // go through the complete list and check the request
   vftr_request_list_t *current_request = vftr_open_request_list;
   while (current_request != NULL) {
      // Test if the current request is still ongoing or completed
      // Note: MPI_Test is a destructive test. It will destroy the request
      //       Thus, running with and without vftr can lead to different program executions
      //       MPI_Request_get_status is a non destructive status check
      MPI_Status tmpStatus;
      int flag;
      PMPI_Request_get_status(current_request->request,
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
         long long tend = vftr_get_runtime_usec ();

         // depending on the type of request
         // it needs to be executed slightly differently
         switch (current_request->communication_type) {
            case vftr_comm_P2P:
               vftr_clear_completed_P2P_request(current_request,
                                                tmpStatus,
                                                tend);
               break;
            case vftr_comm_onesided:
               vftr_clear_completed_onesided_request(current_request,
                                                     tend);
               break;
            case vftr_comm_collective:
               vftr_clear_completed_collective_request(current_request,
                                                       tend);
               break;
            default:
               break;
         }

         // remove the request from the open request list
         // first free request internal memory;
         free(current_request->count);
         free(current_request->type);
         free(current_request->type_idx);
         free(current_request->type_size);
         free(current_request->rank);
         // create a temporary pointer to the current element to be used for deallocation
         vftr_request_list_t *tmp_current_request = current_request;
         // connect the previous element of the list with the next one
         if (current_request->prev == NULL) {
            if (current_request->next == NULL) {
               // only entry in the list
               vftr_open_request_list = NULL;
            } else {
               // first element in the list
               vftr_open_request_list = current_request->next;
               current_request->next->prev = NULL;
            }
         } else {
            if (current_request->next == NULL) {
               // last element in the list
               current_request->prev->next = NULL;
            } else {
               // somewhere in the middle of the list
               current_request->prev->next = current_request->next;
               current_request->next->prev = current_request->prev;
            }
         }
         // advance in list
         current_request = current_request->next;
         // deallocate obsolete entry
         free(tmp_current_request);
         tmp_current_request = NULL;
      } else {
         // advance in list
         current_request = current_request->next;
      }
   } // end of while loop
}

void vftr_clear_completed_P2P_request(vftr_request_list_t *vftr_request,
                                      MPI_Status status, long long tend){
   // extract rank and tag from the completed communication status
   // (if necessary) this is to avoid errors with wildcard usage
   if (vftr_request->tag == MPI_ANY_TAG) {
      vftr_request->tag = status.MPI_TAG;
   }
   // This should only apply for point2point communication
   if (vftr_request->rank[0] == MPI_ANY_SOURCE) {
      vftr_request->rank[0] = status.MPI_SOURCE;
      // check if the communicator is an intercommunicator
      int isintercom;
      PMPI_Comm_test_inter(vftr_request->comm, &isintercom);
      if (isintercom) {
         vftr_request->rank[0] =
            vftr_remote2global_rank(vftr_request->comm,
                                    vftr_request->rank[0]);
      } else {
         vftr_request->rank[0] =
            vftr_local2global_rank(vftr_request->comm,
                                   vftr_request->rank[0]);
      }
   }
   // Get the actual amount of transferred data if it is a receive operation
   // only if it was a point2point communication
   if (vftr_request->dir == recv) {
      int tmpcount;
      PMPI_Get_count(&status, vftr_request->type[0], &tmpcount);
      if (tmpcount != MPI_UNDEFINED) {
         vftr_request->count[0] = tmpcount;
      }
   }
   // store the completed communication info to the outfile
   vftr_store_message_info(vftr_request->dir,
                           vftr_request->count[0],
                           vftr_request->type_idx[0],
                           vftr_request->type_size[0],
                           vftr_request->rank[0],
                           vftr_request->tag,
                           vftr_request->tstart,
                           tend);
}

void vftr_clear_completed_onesided_request(vftr_request_list_t *vftr_request,
                                           long long tend) {
   // Every rank should already be translated to the global rank
   // by the register routine
   vftr_store_message_info(vftr_request->dir,
                           vftr_request->count[0],
                           vftr_request->type_idx[0],
                           vftr_request->type_size[0],
                           vftr_request->rank[0],
                           vftr_request->tag,
                           vftr_request->tstart,
                           tend);
}


void vftr_clear_completed_collective_request(vftr_request_list_t *vftr_request,
                                             long long tend) {
   // Every rank should already be translated to the global rank
   // by the register routine
   for (int i=0; i<vftr_request->nmsg; i++) {
      vftr_store_message_info(vftr_request->dir,
                              vftr_request->count[i],
                              vftr_request->type_idx[i],
                              vftr_request->type_size[i],
                              vftr_request->rank[i],
                              vftr_request->tag,
                              vftr_request->tstart,
                              tend);
   }
}


#endif
