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

#include "vftr_timer.h"
#include "vftr_sync_messages.h"
#include "vftr_mpi_environment.h"
#include "vftr_mpi_buf_addr_const.c"

int vftr_MPI_Allgather(const void *sendbuf, int sendcount,
                       MPI_Datatype sendtype, void *recvbuf, int recvcount,
                       MPI_Datatype recvtype, MPI_Comm comm) {

   // disable profiling based on the Pcontrol level
   if (vftrace_Pcontrol_level == 0) {
      return PMPI_Allgather(sendbuf, sendcount, sendtype, recvbuf,
                            recvcount, recvtype, comm);
   } else {
      long long tstart = vftr_get_runtime_usec();
      int retVal = PMPI_Allgather(sendbuf, sendcount, sendtype, recvbuf,
                                  recvcount, recvtype, comm);
      long long tend = vftr_get_runtime_usec();

      // determine if inter or intra communicator
      int isintercom;
      PMPI_Comm_test_inter(comm, &isintercom);
      if (isintercom) {
         // TODO: handle intercom
         ;
      } else {
         int size;
         PMPI_Comm_size(comm, &size);
         // if sendbuf is special address MPI_IN_PLACE
         // sendcount and sendtype are ignored.
         // Use recvcount and recvtype for statistics
         if (vftr_is_C_MPI_IN_PLACE(sendbuf)) {
            sendcount = recvcount;
            sendtype = recvtype;
         }
         for (int i=0; i<size; i++) {
            vftr_store_sync_message_info(send, sendcount, sendtype, i, -1, comm,
                                         tstart, tend);
            vftr_store_sync_message_info(recv, recvcount, recvtype, i, -1, comm,
                                         tstart, tend);
         }
      }

      return retVal;
   }
}

int vftr_MPI_Allgatherv(const void *sendbuf, int sendcount,
                        MPI_Datatype sendtype, void *recvbuf,
                        const int *recvcounts, const int *displs,
                        MPI_Datatype recvtype, MPI_Comm comm) {

   // disable profiling based on the Pcontrol level
   if (vftrace_Pcontrol_level == 0) {
      return PMPI_Allgatherv(sendbuf, sendcount, sendtype, recvbuf,
                             recvcounts, displs, recvtype, comm);
   } else {
      long long tstart = vftr_get_runtime_usec();
      int retVal = PMPI_Allgatherv(sendbuf, sendcount, sendtype, recvbuf,
                                   recvcounts, displs, recvtype, comm);
      long long tend = vftr_get_runtime_usec();
  
      // determine if inter or intra communicator
      int isintercom;
      PMPI_Comm_test_inter(comm, &isintercom);
      if (isintercom) {
         // TODO: handle intercom
         ;
      } else {
         int size;
         PMPI_Comm_size(comm, &size);
         int rank;
         PMPI_Comm_rank(comm, &rank);
         // if sendbuf is special address MPI_IN_PLACE
         // sendcount and sendtype are ignored.
         // Use recvcount and recvtype for statistics
         if (vftr_is_C_MPI_IN_PLACE(sendbuf)) {
            sendcount = recvcounts[rank];
            sendtype = recvtype;
         }
         for (int i=0; i<size; i++) {
            vftr_store_sync_message_info(send, sendcount, sendtype, i, -1,
                                         comm, tstart, tend);
            vftr_store_sync_message_info(recv, recvcounts[i], recvtype, i, -1,
                                         comm, tstart, tend);
         }
      }
  
      return retVal;
   }
}

int vftr_MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                       MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {

   // disable profiling based on the Pcontrol level
   if (vftrace_Pcontrol_level == 0) {
      return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
   } else {
      long long tstart = vftr_get_runtime_usec();
      int retVal = PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
      long long tend = vftr_get_runtime_usec();
  
      // determine if inter or intra communicator
      int isintercom;
      PMPI_Comm_test_inter(comm, &isintercom);
      if (isintercom) {
         // TODO: handle intercom
         ;
      } else {
         int size;
         PMPI_Comm_size(comm, &size);
         for (int i=0; i<size; i++) {
            vftr_store_sync_message_info(send, count, datatype, i, -1,
                                         comm, tstart, tend);
            vftr_store_sync_message_info(recv, count, datatype, i, -1,
                                         comm, tstart, tend);
         }
      }
  
      return retVal;
   }
}

int vftr_MPI_Alltoall(const void *sendbuf, int sendcount,
                      MPI_Datatype sendtype, void *recvbuf, int recvcount,
                      MPI_Datatype recvtype, MPI_Comm comm) {

   // disable profiling based on the Pcontrol level
   if (vftrace_Pcontrol_level == 0) {
      return PMPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf,
                           recvcount, recvtype, comm);
   } else {
      long long tstart = vftr_get_runtime_usec();
      int retVal = PMPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf,
                                 recvcount, recvtype, comm);
      long long tend = vftr_get_runtime_usec();
  
      // determine if inter or intra communicator
      int isintercom;
      PMPI_Comm_test_inter(comm, &isintercom);
      if (isintercom) {
         // TODO: handle intercom
         ;
      } else {
         int size;
         PMPI_Comm_size(comm, &size);
         // if sendbuf is special address MPI_IN_PLACE
         // sendcount and sendtype are ignored.
         // Use recvcount and recvtype for statistics
         if (vftr_is_C_MPI_IN_PLACE(sendbuf)) {
            sendcount = recvcount;
            sendtype = recvtype;
         }
         for (int i=0; i<size; i++) {
            vftr_store_sync_message_info(send, sendcount, sendtype, i, 0,
                                         comm, tstart, tend);
            vftr_store_sync_message_info(recv, recvcount, recvtype, i, 0,
                                         comm, tstart, tend);
         }
      }
      return retVal;
   }
}

int vftr_MPI_Alltoallv(const void *sendbuf, const int *sendcounts,
                       const int *sdispls, MPI_Datatype sendtype, void *recvbuf,
                       const int *recvcounts, const int *rdispls,
                       MPI_Datatype recvtype, MPI_Comm comm) {

   // disable profiling based on the Pcontrol level
   if (vftrace_Pcontrol_level == 0) {
      return PMPI_Alltoallv(sendbuf, sendcounts, sdispls, sendtype,
                            recvbuf, recvcounts, rdispls, recvtype, comm);
   } else {
      long long tstart = vftr_get_runtime_usec();
      int retVal = PMPI_Alltoallv(sendbuf, sendcounts, sdispls, sendtype,
                                  recvbuf, recvcounts, rdispls, recvtype, comm);
      long long tend = vftr_get_runtime_usec();

      // determine if inter or intra communicator
      int isintercom;
      PMPI_Comm_test_inter(comm, &isintercom);
      if (isintercom) {
         // TODO: handle intercom
         ;
      } else {
         int size;
         PMPI_Comm_size(comm, &size);
         // if sendbuf is special address MPI_IN_PLACE
         // sendcount and sendtype are ignored.
         // Use recvcount and recvtype for statistics
         if (vftr_is_C_MPI_IN_PLACE(sendbuf)) {
            sendcounts = recvcounts;
            sendtype = recvtype;
         }
         for (int i=0; i<size; i++) {
            vftr_store_sync_message_info(send, sendcounts[i], sendtype, i, 0,
                                         comm, tstart, tend);
            vftr_store_sync_message_info(recv, recvcounts[i], recvtype, i, 0,
                                         comm, tstart, tend);
         }
      }

      return retVal;
   }
}

int vftr_MPI_Alltoallw(const void *sendbuf, const int *sendcounts,
                       const int *sdispls, const MPI_Datatype *sendtypes,
                       void *recvbuf, const int *recvcounts, const int *rdispls,
                       const MPI_Datatype *recvtypes, MPI_Comm comm) {

   // disable profiling based on the Pcontrol level
   if (vftrace_Pcontrol_level == 0) {
      return PMPI_Alltoallw(sendbuf, sendcounts, sdispls, sendtypes,
                            recvbuf, recvcounts, rdispls, recvtypes, comm);
   } else {
      long long tstart = vftr_get_runtime_usec();
      int retVal = PMPI_Alltoallw(sendbuf, sendcounts, sdispls, sendtypes,
                                  recvbuf, recvcounts, rdispls, recvtypes, comm);
      long long tend = vftr_get_runtime_usec();

      // determine if inter or intra communicator
      int isintercom;
      PMPI_Comm_test_inter(comm, &isintercom);
      if (isintercom) {
         // TODO: handle intercom
         ;
      } else {
         int size;
         PMPI_Comm_size(comm, &size);
         // if sendbuf is special address MPI_IN_PLACE
         // sendcount and sendtype are ignored.
         // Use recvcount and recvtype for statistics
         if (vftr_is_C_MPI_IN_PLACE(sendbuf)) {
            sendcounts = recvcounts;
            sendtypes = recvtypes;
         }
         for (int i=0; i<size; i++) {
            vftr_store_sync_message_info(send, sendcounts[i], sendtypes[i],
                                         i, 0, comm, tstart, tend);
            vftr_store_sync_message_info(recv, recvcounts[i], recvtypes[i],
                                         i, 0, comm, tstart, tend);
         }
      }

      return retVal;
   }
}

int vftr_MPI_Bcast(void *buffer, int count, MPI_Datatype datatype,
                   int root, MPI_Comm comm) {

   // disable profiling based on the Pcontrol level
   if (vftrace_Pcontrol_level == 0) {
      return PMPI_Bcast(buffer, count, datatype, root, comm);
   } else {
      long long tstart = vftr_get_runtime_usec();
      int retVal = PMPI_Bcast(buffer, count, datatype, root, comm);
      long long tend = vftr_get_runtime_usec();

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
            for (int i=0; i<size; i++) {
               // translate the i-th rank in group B to the global rank
               int global_peer_rank = vftr_remote2global_rank(comm, i);
               // store message info with MPI_COMM_WORLD as communicator
               // to prevent additional (and thus faulty rank translation)
               vftr_store_sync_message_info(recv, count, datatype,
                                            global_peer_rank, -1, MPI_COMM_WORLD,
                                            tstart, tend);
            }
         } else if (root == MPI_PROC_NULL) {
            // All other processes from group A pass wildcard MPI_PROC NULL
            // They do not participate in intercommunicator gather
            ;
         } else {
            // All other processes must be located in group B
            // root is the rank-id in group A Therefore no problems with 
            // rank translation should arise
            vftr_store_sync_message_info(send, count, datatype,
                                         root, -1, comm, tstart, tend);
         }
      } else {
         // in intracommunicators the expected behaviour is to
         // gather from root to all other processes in the communicator
         int rank;
         PMPI_Comm_rank(comm, &rank);
         if (rank == root) {
            int size;
            PMPI_Comm_size(comm, &size);
            for (int i=0; i<size; i++) {
               vftr_store_sync_message_info(recv, count, datatype, i, -1,
                                            comm, tstart, tend);
            }
         } else {
            vftr_store_sync_message_info(send, count, datatype, root, -1,
                                         comm, tstart, tend);
         }
      }

      return retVal;
   }
}

int vftr_MPI_Gather(const void *sendbuf, int sendcount,
                    MPI_Datatype sendtype, void *recvbuf, int recvcount,
                    MPI_Datatype recvtype, int root, MPI_Comm comm) {

   // disable profiling based on the Pcontrol level
   if (vftrace_Pcontrol_level == 0) {
      return PMPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount,
                         recvtype, root, comm);
   } else {
      long long tstart = vftr_get_runtime_usec();
      int retVal = PMPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount,
                               recvtype, root, comm);
      long long tend = vftr_get_runtime_usec();

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
            for (int i=0; i<size; i++) {
               // translate the i-th rank in group B to the global rank
               int global_peer_rank = vftr_remote2global_rank(comm, i);
               // store message info with MPI_COMM_WORLD as communicator
               // to prevent additional (and thus faulty rank translation)
               vftr_store_sync_message_info(recv, recvcount, recvtype,
                                            global_peer_rank, -1, MPI_COMM_WORLD,
                                            tstart, tend);
            }
         } else if (root == MPI_PROC_NULL) {
            // All other processes from group A pass wildcard MPI_PROC NULL
            // They do not participate in intercommunicator bcasts
            ;
         } else {
            // All other processes must be located in group B
            // root is the rank-id in group A Therefore no problems with
            // rank translation should arise
            vftr_store_sync_message_info(send, sendcount, sendtype,
                                         root, -1, comm, tstart, tend);
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
               sendcount = recvcount;
               sendtype = recvtype;
            }
            // self communication of root process
            vftr_store_sync_message_info(send, sendcount, sendtype,
                                         root, -1, comm, tstart, tend);
            for (int i=0; i<size; i++) {
               vftr_store_sync_message_info(recv, recvcount, recvtype,
                                            i, -1, comm, tstart, tend);
            }
         } else {
            vftr_store_sync_message_info(send, sendcount, sendtype,
                                         root, -1, comm, tstart, tend);
         }
      }

      return retVal;
   }
}

int vftr_MPI_Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                     void *recvbuf, const int *recvcounts, const int *displs,
                     MPI_Datatype recvtype, int root, MPI_Comm comm) {

   // disable profiling based on the Pcontrol level
   if (vftrace_Pcontrol_level == 0) {
      return PMPI_Gatherv(sendbuf, sendcount, sendtype, recvbuf,
                          recvcounts, displs, recvtype, root, comm);
   } else {
      long long tstart = vftr_get_runtime_usec();
      int retVal = PMPI_Gatherv(sendbuf, sendcount, sendtype, recvbuf,
                                recvcounts, displs, recvtype, root, comm);
      long long tend = vftr_get_runtime_usec();
  
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
            MPI_Comm_remote_size(comm, &size);
            for (int i=0; i<size; i++) {
               // translate the i-th rank in group B to the global rank
               int global_peer_rank = vftr_remote2global_rank(comm, i);
               // store message info with MPI_COMM_WORLD as communicator
               // to prevent additional (and thus faulty rank translation)
               vftr_store_sync_message_info(recv, recvcounts[i], recvtype,
                                            global_peer_rank, -1, MPI_COMM_WORLD,
                                            tstart, tend);
            }
         } else if (root == MPI_PROC_NULL) {
            // All other processes from group A pass wildcard MPI_PROC NULL
            // They do not participate in intercommunicator bcasts
            ;
         } else {
            // All other processes must be located in group B
            // root is the rank-id in group A Therefore no problems with
            // rank translation should arise
            vftr_store_sync_message_info(send, sendcount, sendtype,
                                         root, -1, comm, tstart, tend);
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
               sendcount = recvcounts[rank];
               sendtype = recvtype;
            }
            vftr_store_sync_message_info(send, sendcount, sendtype,
                                         root, -1, comm, tstart, tend);
            for (int i=0; i<size; i++) {
               vftr_store_sync_message_info(recv, recvcounts[i], recvtype,
                                            i, -1, comm, tstart, tend);
            }
         } else {
            vftr_store_sync_message_info(send, sendcount, sendtype, root,
                                         -1, comm, tstart, tend);
         }
      }
  
      return retVal;
   }
}

int vftr_MPI_Reduce(const void *sendbuf, void *recvbuf, int count,
                    MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm) {

   // disable profiling based on the Pcontrol level
   if (vftrace_Pcontrol_level == 0) {
      return PMPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
   } else {
      long long tstart = vftr_get_runtime_usec();
      int retVal = PMPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
      long long tend = vftr_get_runtime_usec();

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
            for (int i=0; i<size; i++) {
               // translate the i-th rank in group B to the global rank
               int global_peer_rank = vftr_remote2global_rank(comm, i);
               // store message info with MPI_COMM_WORLD as communicator
               // to prevent additional (and thus faulty rank translation)
               vftr_store_sync_message_info(recv, count, datatype,
                                            global_peer_rank, -1, MPI_COMM_WORLD,
                                            tstart, tend);
            }
         } else if (root == MPI_PROC_NULL) {
            // All other processes from group A pass wildcard MPI_PROC NULL
            // They do not participate in intercommunicator bcasts
            ;
         } else {
            // All other processes must be located in group B
            // root is the rank-id in group A Therefore no problems with 
            // rank translation should arise
            vftr_store_sync_message_info(send, count, datatype,
                                         root, -1, comm, tstart, tend);
         }
      } else {
         // in intracommunicators the expected behaviour is to
         // bcast from root to all other processes in the communicator
         int rank;
         PMPI_Comm_rank(comm, &rank);
         if (rank == root) {
            int size;
            PMPI_Comm_size(comm, &size);
            for (int i=0; i<size; i++) {
               vftr_store_sync_message_info(recv, count, datatype,
                                            i, -1, comm, tstart, tend);
            }
            // self communication
            vftr_store_sync_message_info(send, count, datatype,
                                         rank, -1, comm, tstart, tend);
         } else {
            vftr_store_sync_message_info(send, count, datatype,
                                         root, -1, comm, tstart, tend);
         }
      }

      return retVal;
   }
}

int vftr_MPI_Reduce_scatter(const void *sendbuf, void *recvbuf,
                            const int *recvcounts, MPI_Datatype datatype,
                            MPI_Op op, MPI_Comm comm) {

   // disable profiling based on the Pcontrol level
   if (vftrace_Pcontrol_level == 0) {
      return PMPI_Reduce_scatter(sendbuf, recvbuf, recvcounts, datatype, op, comm);
   } else {
      // TODO
      int retVal = PMPI_Reduce_scatter(sendbuf, recvbuf, recvcounts, datatype, op, comm);

      return retVal;
   }
}

int vftr_MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                     void *recvbuf, int recvcount, MPI_Datatype recvtype,
                     int root, MPI_Comm comm) {

   // disable profiling based on the Pcontrol level
   if (vftrace_Pcontrol_level == 0) {
      return PMPI_Scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount,
                          recvtype, root, comm);
   } else {
      long long tstart = vftr_get_runtime_usec();
      int retVal = PMPI_Scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount,
                                recvtype, root, comm);
      long long tend = vftr_get_runtime_usec();
  
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
            for (int i=0; i<size; i++) {
               // translate the i-th rank in group B to the global rank
               int global_peer_rank = vftr_remote2global_rank(comm, i);
               // store message info with MPI_COMM_WORLD as communicator
               // to prevent additional (and thus faulty rank translation)
               vftr_store_sync_message_info(send, sendcount, sendtype,
                                            global_peer_rank, -1, MPI_COMM_WORLD,
                                            tstart, tend);
            }
         } else if (root == MPI_PROC_NULL) {
            // All other processes from group A pass wildcard MPI_PROC NULL
            // They do not participate in intercommunicator bcasts
            ;
         } else {
            // All other processes must be located in group B
            // root is the rank-id in group A Therefore no problems with
            // rank translation should arise
            vftr_store_sync_message_info(recv, recvcount, recvtype,
                                         root, -1, comm, tstart, tend);
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
            if (vftr_is_C_MPI_IN_PLACE(sendbuf)) {
               recvcount = sendcount;
               recvtype = sendtype;
            }
            for (int i=0; i<size; i++) {
               vftr_store_sync_message_info(send, sendcount, sendtype,
                                            i, -1, comm, tstart, tend);
            }
            // self communication
            vftr_store_sync_message_info(recv, recvcount, recvtype,
                                         root, -1, comm, tstart, tend);
         } else {
            vftr_store_sync_message_info(recv, recvcount, recvtype, root,
                                         -1, comm, tstart, tend);
         }
      }
  
      return retVal;
   }
}

int vftr_MPI_Scatterv(const void *sendbuf, const int *sendcounts,
                      const int *displs, MPI_Datatype sendtype,
                      void *recvbuf, int recvcount, MPI_Datatype recvtype,
                      int root, MPI_Comm comm) {

   // disable profiling based on the Pcontrol level
   if (vftrace_Pcontrol_level == 0) {
      return PMPI_Scatterv(sendbuf, sendcounts, displs, sendtype,
                           recvbuf, recvcount, recvtype, root, comm);
   } else {
      long long tstart = vftr_get_runtime_usec();
      int retVal = PMPI_Scatterv(sendbuf, sendcounts, displs, sendtype,
                                 recvbuf, recvcount, recvtype, root, comm);
      long long tend = vftr_get_runtime_usec();
  
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
            for (int i=0; i<size; i++) {
               // translate the i-th rank in group B to the global rank
               int global_peer_rank = vftr_remote2global_rank(comm, i);
               // store message info with MPI_COMM_WORLD as communicator
               // to prevent additional (and thus faulty rank translation)
               vftr_store_sync_message_info(send, sendcounts[i], sendtype,
                                            global_peer_rank, -1, MPI_COMM_WORLD,
                                            tstart, tend);
            }
         } else if (root == MPI_PROC_NULL) {
            // All other processes from group A pass wildcard MPI_PROC NULL
            // They do not participate in intercommunicator bcasts
            ;
         } else {
            // All other processes must be located in group B
            // root is the rank-id in group A Therefore no problems with
            // rank translation should arise
            vftr_store_sync_message_info(recv, recvcount, recvtype,
                                         root, -1, comm, tstart, tend);
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
            if (vftr_is_C_MPI_IN_PLACE(sendbuf)) {
               recvcount = sendcounts[rank];
               recvtype = sendtype;
            }
            for (int i=0; i<size; i++) {
               vftr_store_sync_message_info(send, sendcounts[i], sendtype,
                                            i, -1, comm, tstart, tend);
            }
            // self communication
            vftr_store_sync_message_info(recv, recvcount, recvtype,
                                         rank, -1, comm, tstart, tend);
         } else {
            vftr_store_sync_message_info(recv, recvcount, recvtype,
                                         root, -1, comm, tstart, tend);
         }
      }
  
      return retVal;
   }
}

#endif
