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

#include "vftr_timer.h"
#include "vftr_collective_requests.h"
#include "vftr_mpi_utils.h"
#include "vftr_mpi_buf_addr_const.h"

int vftr_MPI_Iscatterv(const void *sendbuf, const int *sendcounts,
                       const int *displs, MPI_Datatype sendtype,
                       void *recvbuf, int recvcount, MPI_Datatype recvtype,
                       int root, MPI_Comm comm, MPI_Request *request) {
   long long tstart = vftr_get_runtime_usec();
   int retVal = PMPI_Iscatterv(sendbuf, sendcounts, displs, sendtype,
                              recvbuf, recvcount, recvtype, root, comm, request);

   long long t2start = vftr_get_runtime_usec();
   // determine if inter or intra communicator
   int isintercom;
   PMPI_Comm_test_inter(comm, &isintercom);
   if (isintercom) {
      // in intercommunicators the behaviour is more complicated
      // There are two groups A and B
      // In group A the root process is located.
      if (root == MPI_ROOT) {
         // The root process get the special process wildcard MPI_ROOT
         // get the size of group B
         int size;
         PMPI_Comm_remote_size(comm, &size);
         int *tmpsendcounts = (int*) malloc(sizeof(int)*size);
         MPI_Datatype *tmpsendtype = (MPI_Datatype*) malloc(sizeof(MPI_Datatype)*size);
         int *tmppeer_ranks= (int*) malloc(sizeof(int)*size);
         // messages to be received
         for (int i=0; i<size; i++) {
            tmpsendcounts[i] = sendcounts[i];
            tmpsendtype[i] = sendtype;
            // translate the i-th rank in group B to the global rank
            tmppeer_ranks[i] = vftr_remote2global_rank(comm, i);
         }
         // Store tmp-pointers for delayed deallocation
         int n_tmp_ptr = 2;
         void **tmp_ptrs = (void**) malloc(n_tmp_ptr*sizeof(void*));
         tmp_ptrs[0] = (void*) sendcounts;
         tmp_ptrs[1] = (void*) displs;
         // Register request with MPI_COMM_WORLD as communicator
         // to prevent additional (and thus faulty rank translation)
         vftr_register_collective_request(send, size, tmpsendcounts, tmpsendtype,
                                          tmppeer_ranks, MPI_COMM_WORLD,
                                          *request, n_tmp_ptr, tmp_ptrs, tstart);
         // cleanup temporary arrays
         free(tmpsendcounts);
         tmpsendcounts = NULL;
         free(tmpsendtype);
         tmpsendtype = NULL;
         free(tmppeer_ranks);
         tmppeer_ranks = NULL;
      } else if (root == MPI_PROC_NULL) {
         // All other processes from group A pass wildcard MPI_PROC NULL
         // They do not participate in intercommunicator bcasts
         ;
      } else {
         // All other processes must be located in group B
         // root is the rank-id in group A Therefore no problems with
         // rank translation should arise
         vftr_register_collective_request(recv, 1, &recvcount, &recvtype, &root,
                                          comm, *request, 0, NULL, tstart);
      }
   } else {
      // in intracommunicators the expected behaviour is to
      // bcast from root to all other processes in the communicator
      int rank;
      PMPI_Comm_rank(comm, &rank);
      if (rank == root) {
         int size;
         PMPI_Comm_size(comm, &size);
         // if recvbuf is special address MPI_IN_PLACE
         // recvcount and recvtype are ignored.
         // Use sendcount and sendtype for statistics
         if (vftr_is_C_MPI_IN_PLACE(recvbuf)) {
            if (size > 1) {
               // For the in-place option no self communication is executed

               // allocate memory for the temporary arrays
               // to register communication request
               int *tmpsendcounts = (int*) malloc(sizeof(int)*(size-1));
               MPI_Datatype *tmpsendtype = (MPI_Datatype*) malloc(sizeof(MPI_Datatype)*(size-1));
               int *tmppeer_ranks= (int*) malloc(sizeof(int)*(size-1));
               // messeges to be send
               int idx = 0;
               for (int i=0; i<rank; i++) {
                  tmpsendcounts[idx] = sendcounts[i];
                  tmpsendtype[idx] = sendtype;
                  tmppeer_ranks[idx] = i;
                  idx++;
               }
               for (int i=rank+1; i<size; i++) {
                  tmpsendcounts[idx] = sendcounts[i];
                  tmpsendtype[idx] = sendtype;
                  tmppeer_ranks[idx] = i;
                  idx++;
               }
               // Store tmp-pointers for delayed deallocation
               int n_tmp_ptr = 2;
               void **tmp_ptrs = (void**) malloc(n_tmp_ptr*sizeof(void*));
               tmp_ptrs[0] = (void*) sendcounts;
               tmp_ptrs[1] = (void*) displs;
               vftr_register_collective_request(send, size-1, tmpsendcounts, tmpsendtype,
                                                tmppeer_ranks, comm, *request,
                                                n_tmp_ptr, tmp_ptrs, tstart);
               // cleanup temporary arrays
               free(tmpsendcounts);
               tmpsendcounts = NULL;
               free(tmpsendtype);
               tmpsendtype = NULL;
               free(tmppeer_ranks);
               tmppeer_ranks = NULL;
            }
         } else {
            // allocate memory for the temporary arrays
            // to register communication request
            int *tmpsendcounts = (int*) malloc(sizeof(int)*size);
            MPI_Datatype *tmpsendtype = (MPI_Datatype*) malloc(sizeof(MPI_Datatype)*size);
            int *tmppeer_ranks= (int*) malloc(sizeof(int)*size);
            // messeges to be send
            for (int i=0; i<size; i++) {
               tmpsendcounts[i] = sendcounts[i];
               tmpsendtype[i] = sendtype;
               tmppeer_ranks[i] = i;
            }
            // Store tmp-pointers for delayed deallocation
            int n_tmp_ptr = 2;
            void **tmp_ptrs = (void**) malloc(n_tmp_ptr*sizeof(void*));
            tmp_ptrs[0] = (void*) sendcounts;
            tmp_ptrs[1] = (void*) displs;
            vftr_register_collective_request(send, size, tmpsendcounts, tmpsendtype,
                                             tmppeer_ranks, comm, *request,
                                             n_tmp_ptr, tmp_ptrs, tstart);
            // cleanup temporary arrays
            free(tmpsendcounts);
            tmpsendcounts = NULL;
            free(tmpsendtype);
            tmpsendtype = NULL;
            free(tmppeer_ranks);
            tmppeer_ranks = NULL;
   
            // self communication of root process
            vftr_register_collective_request(recv, 1, &recvcount, &recvtype, &root,
                                             comm, *request, 0, NULL, tstart);
         }
      } else {
         vftr_register_collective_request(recv, 1, &recvcount, &recvtype, &root,
                                          comm, *request, 0, NULL, tstart);
      }
   }
   long long t2end = vftr_get_runtime_usec();

   vftr_mpi_overhead_usec += t2end - t2start;
  
   return retVal;
}

#endif
