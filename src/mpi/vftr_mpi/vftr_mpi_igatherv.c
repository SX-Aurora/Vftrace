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

int vftr_MPI_Igatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                      void *recvbuf, const int *recvcounts, const int *displs,
                      MPI_Datatype recvtype, int root, MPI_Comm comm,
                      MPI_Request *request) {
   long long tstart = vftr_get_runtime_usec();
   int retVal = PMPI_Igatherv(sendbuf, sendcount, sendtype, recvbuf,
                              recvcounts, displs, recvtype, root, comm, request);

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
         int *tmprecvcounts = (int*) malloc(sizeof(int)*size);
         MPI_Datatype *tmprecvtype = (MPI_Datatype*) malloc(sizeof(MPI_Datatype)*size);
         int *tmppeer_ranks= (int*) malloc(sizeof(int)*size);
         // messages to be received
         for (int i=0; i<size; i++) {
            tmprecvcounts[i] = recvcounts[i];
            tmprecvtype[i] = recvtype;
            // translate the i-th rank in group B to the global rank
            tmppeer_ranks[i] = vftr_remote2global_rank(comm, i);
         }
         // Register request with MPI_COMM_WORLD as communicator
         // to prevent additional (and thus faulty rank translation)
         vftr_register_collective_request(recv, size, tmprecvcounts, tmprecvtype,
                                          tmppeer_ranks, MPI_COMM_WORLD,
                                          *request, tstart);
         // cleanup temporary arrays
         free(tmprecvcounts);
         tmprecvcounts = NULL;
         free(tmprecvtype);
         tmprecvtype = NULL;
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
         vftr_register_collective_request(send, 1, &sendcount, &sendtype, &root,
                                          comm, *request, tstart);
      }
   } else {
      // in intracommunicators the expected behaviour is to
      // bcast from root to all other processes in the communicator
      int rank;
      PMPI_Comm_rank(comm, &rank);
      if (rank == root) {
         int size;
         PMPI_Comm_size(comm, &size);
         // if sendbuf is special address MPI_IN_PLACE
         // sendcount and sendtype are ignored.
         // Use recvcount and recvtype for statistics
         if (vftr_is_C_MPI_IN_PLACE(sendbuf)) {
            if (size > 1) {
               // For the in-place option no self communication is executed
               // allocate memory for the temporary arrays
               // to register communication request
               int *tmprecvcounts = (int*) malloc(sizeof(int)*(size-1));
               MPI_Datatype *tmprecvtype = (MPI_Datatype*) malloc(sizeof(MPI_Datatype)*(size-1));
               int *tmppeer_ranks= (int*) malloc(sizeof(int)*(size-1));
               // messeges to be received
               int idx = 0;
               for (int i=0; i<rank; i++) {
                  tmprecvcounts[idx] = recvcounts[i];
                  tmprecvtype[idx] = recvtype;
                  tmppeer_ranks[idx] = i;
                  idx++;
               }
               for (int i=rank+1; i<size; i++) {
                  tmprecvcounts[idx] = recvcounts[i];
                  tmprecvtype[idx] = recvtype;
                  tmppeer_ranks[idx] = i;
                  idx++;
               }
               vftr_register_collective_request(recv, size-1, tmprecvcounts, tmprecvtype,
                                                tmppeer_ranks, comm, *request, tstart);
               // cleanup temporary arrays
               free(tmprecvcounts);
               tmprecvcounts = NULL;
               free(tmprecvtype);
               tmprecvtype = NULL;
               free(tmppeer_ranks);
               tmppeer_ranks = NULL;
            }
         } else {
            // self communication of root process
            vftr_register_collective_request(send, 1, &sendcount, &sendtype, &root,
                                             comm, *request, tstart);
   
            // allocate memory for the temporary arrays
            // to register communication request
            int *tmprecvcounts = (int*) malloc(sizeof(int)*size);
            MPI_Datatype *tmprecvtype = (MPI_Datatype*) malloc(sizeof(MPI_Datatype)*size);
            int *tmppeer_ranks= (int*) malloc(sizeof(int)*size);
            // messeges to be received
            for (int i=0; i<size; i++) {
               tmprecvcounts[i] = recvcounts[i];
               tmprecvtype[i] = recvtype;
               tmppeer_ranks[i] = i;
            }
            vftr_register_collective_request(recv, size, tmprecvcounts, tmprecvtype,
                                             tmppeer_ranks, comm, *request, tstart);
            // cleanup temporary arrays
            free(tmprecvcounts);
            tmprecvcounts = NULL;
            free(tmprecvtype);
            tmprecvtype = NULL;
            free(tmppeer_ranks);
            tmppeer_ranks = NULL;
         }
      } else {
         vftr_register_collective_request(send, 1, &sendcount, &sendtype, &root,
                                          comm, *request, tstart);
      }
   }
   long long t2end = vftr_get_runtime_usec();

   vftr_mpi_overhead_usec += t2end - t2start;

   return retVal;
   
}

#endif
