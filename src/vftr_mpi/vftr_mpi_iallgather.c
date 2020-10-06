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
#include "vftr_async_messages.h"
#include "vftr_collective_requests.h"
#include "vftr_mpi_utils.h"
#include "vftr_mpi_buf_addr_const.h"

int vftr_MPI_Iallgather(const void *sendbuf, int sendcount,
                        MPI_Datatype sendtype, void *recvbuf, int recvcount,
                        MPI_Datatype recvtype, MPI_Comm comm,
                        MPI_Request *request) {

   // disable profiling based on the Pcontrol level
   if (vftr_no_mpi_logging()) {
      return PMPI_Iallgather(sendbuf, sendcount, sendtype, recvbuf,
                             recvcount, recvtype, comm, request);
   } else {
      long long tstart = vftr_get_runtime_usec();
      int retVal = PMPI_Iallgather(sendbuf, sendcount, sendtype, recvbuf,
                                   recvcount, recvtype, comm, request);

      long long t2start = vftr_get_runtime_usec();
      // determine if inter or intra communicator
      int isintercom;
      PMPI_Comm_test_inter(comm, &isintercom);
      if (isintercom) {
         // Every process of group A sends sendcount data to and
         // receives recvcount data from every process in group B and
         // vice versa
         int size;
         PMPI_Comm_remote_size(comm, &size);
         // allocate memory for the temporary arrays
         // to register communication request
         int *tmpcount = (int*) malloc(sizeof(int)*size);
         MPI_Datatype *tmptype = (MPI_Datatype*) malloc(sizeof(MPI_Datatype)*size);
         int *peer_ranks = (int*) malloc(sizeof(int)*size);
         // messages to be send
         for (int i=0; i<size; i++) {
            tmpcount[i] = sendcount;
            tmptype[i] = sendtype;
            // translate the i-th rank in the remote group to the global rank
            peer_ranks[i] = vftr_remote2global_rank(comm, i); 
         }
         // Register request with MPI_COMM_WORLD as communicator
         // to prevent additional (and thus faulty rank translation)
         vftr_register_collective_request(send, size, tmpcount, tmptype, peer_ranks,
                                          MPI_COMM_WORLD, *request, tstart);
         // messages to be received
         for (int i=0; i<size; i++) {
            tmpcount[i] = recvcount;
            tmptype[i] = recvtype;
            // translate the i-th rank in the remote group to the global rank
            peer_ranks[i] = vftr_remote2global_rank(comm, i); 
         }
         // Register request with MPI_COMM_WORLD as communicator
         // to prevent additional (and thus faulty rank translation)
         vftr_register_collective_request(recv, size, tmpcount, tmptype, peer_ranks,
                                          MPI_COMM_WORLD, *request, tstart);
         // cleanup temporary arrays
         free(tmpcount);
         tmpcount = NULL;
         free(tmptype);
         tmptype = NULL;
         free(peer_ranks);
         peer_ranks = NULL;
      } else {
         int size;
         PMPI_Comm_size(comm, &size);
         // if sendbuf is special address MPI_IN_PLACE
         // sendcount and sendtype are ignored.
         // Use recvcount and recvtype for statistics
         if (vftr_is_C_MPI_IN_PLACE(sendbuf)) {
            if (size > 1) {
               // For the in-place option no self communication is executed
               int rank;
               PMPI_Comm_rank(comm, &rank);

               // allocate memory for the temporary arrays
               // to register communication request
               int *tmpcount = (int*) malloc(sizeof(int)*(size-1));
               MPI_Datatype *tmptype = (MPI_Datatype*) malloc(sizeof(MPI_Datatype)*(size-1));
               int *peer_ranks = (int*) malloc(sizeof(int)*(size-1));
               // messages to be send and received
               int idx = 0;
               for (int i=0; i<rank; i++) {
                  tmpcount[idx] = recvcount;
                  tmptype[idx] = recvtype;
                  peer_ranks[idx] = i;
                  idx++;
               }
               for (int i=rank+1; i<size; i++) {
                  tmpcount[idx] = recvcount;
                  tmptype[idx] = recvtype;
                  peer_ranks[idx] = i;
                  idx++;
               }
               // sendcount and sendtype are ignored,
               // thus the tmpcount and tmptype array is identical
               vftr_register_collective_request(send, size-1, tmpcount, tmptype, peer_ranks,
                                                comm, *request, tstart);
               vftr_register_collective_request(recv, size-1, tmpcount, tmptype, peer_ranks,
                                                comm, *request, tstart);
               // cleanup temporary arrays
               free(tmpcount);
               tmpcount = NULL;
               free(tmptype);
               tmptype = NULL;
               free(peer_ranks);
               peer_ranks = NULL;
            }
         } else {
            // allocate memory for the temporary arrays
            // to register communication request
            int *tmpcount = (int*) malloc(sizeof(int)*size);
            MPI_Datatype *tmptype = (MPI_Datatype*) malloc(sizeof(MPI_Datatype)*size);
            int *peer_ranks = (int*) malloc(sizeof(int)*size);
            // messages to be send
            for (int i=0; i<size; i++) {
               tmpcount[i] = sendcount;
               tmptype[i] = sendtype;
               peer_ranks[i] = i;
            }
            vftr_register_collective_request(send, size, tmpcount, tmptype, peer_ranks,
                                             comm, *request, tstart);
            // messages to be received
            for (int i=0; i<size; i++) {
               tmpcount[i] = recvcount;
               tmptype[i] = recvtype;
               peer_ranks[i] = i;
            }
            vftr_register_collective_request(recv, size, tmpcount, tmptype, peer_ranks,
                                             comm, *request, tstart);
            // cleanup temporary arrays
            free(tmpcount);
            tmpcount = NULL;
            free(tmptype);
            tmptype = NULL;
            free(peer_ranks);
            peer_ranks = NULL;
         }
      }
      long long t2end = vftr_get_runtime_usec();

      vftr_mpi_overhead_usec += t2end - t2start;

      return retVal;
   }
}

#endif
