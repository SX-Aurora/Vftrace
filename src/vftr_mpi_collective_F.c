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

#include <stdlib.h>

#ifdef _MPI
#include <mpi.h>

#include "vftr_mpi_buf_addr_const.h"
#include "vftr_mpi_collective.h"

void vftr_MPI_Allgather_F(void *sendbuf, MPI_Fint *sendcount, MPI_Fint *f_sendtype,
                          void *recvbuf, MPI_Fint *recvcount, MPI_Fint *f_recvtype,
                          MPI_Fint *f_comm, MPI_Fint *f_error) {

   MPI_Datatype c_sendtype = PMPI_Type_f2c(*f_sendtype);
   MPI_Datatype c_recvtype = PMPI_Type_f2c(*f_recvtype);
   MPI_Comm c_comm = PMPI_Comm_f2c(*f_comm);

   sendbuf = (void*) vftr_is_F_MPI_IN_PLACE(sendbuf) ? MPI_IN_PLACE : sendbuf;
   sendbuf = (void*) vftr_is_F_MPI_BOTTOM(sendbuf) ? MPI_BOTTOM : sendbuf;
   recvbuf = (void*) vftr_is_F_MPI_BOTTOM(recvbuf) ? MPI_BOTTOM : recvbuf;

   int c_error = vftr_MPI_Allgather(sendbuf,
                                    (int)(*sendcount),
                                    c_sendtype,
                                    recvbuf,
                                    (int)(*recvcount),
                                    c_recvtype,
                                    c_comm);

   *f_error = (MPI_Fint) (c_error);
}

void vftr_MPI_Allgatherv_F(void *sendbuf, MPI_Fint *sendcount, MPI_Fint *f_sendtype,
                           void *recvbuf, MPI_Fint *f_recvcounts, MPI_Fint *f_displs, 
                           MPI_Fint *f_recvtype, MPI_Fint *f_comm, MPI_Fint *f_error) {

   MPI_Comm c_comm = PMPI_Comm_f2c(*f_comm);
   int size;
   PMPI_Comm_size(c_comm, &size);
   MPI_Datatype c_sendtype = PMPI_Type_f2c(*f_sendtype);
   int *c_recvcounts = (int*) malloc(size*sizeof(int));
   for (int i=0; i<size; i++) {
      c_recvcounts[i] = (int) f_recvcounts[i];
   }
   int *c_displs = (int*) malloc(size*sizeof(int));
   for (int i=0; i<size; i++) {
      c_displs[i] = (int) f_displs[i];
   }
   MPI_Datatype c_recvtype = PMPI_Type_f2c(*f_recvtype);

   sendbuf = (void*) vftr_is_F_MPI_IN_PLACE(sendbuf) ? MPI_IN_PLACE : sendbuf;
   sendbuf = (void*) vftr_is_F_MPI_BOTTOM(sendbuf) ? MPI_BOTTOM : sendbuf;
   recvbuf = (void*) vftr_is_F_MPI_BOTTOM(recvbuf) ? MPI_BOTTOM : recvbuf;

   int c_error = vftr_MPI_Allgatherv(sendbuf,
                                     (int)(*sendcount),
                                     c_sendtype,
                                     recvbuf,
                                     c_recvcounts,
                                     c_displs,
                                     c_recvtype,
                                     c_comm);

   free(c_recvcounts);
   free(c_displs);

   *f_error = (MPI_Fint) (c_error);
}

void vftr_MPI_Allreduce_F(void *sendbuf, void *recvbuf, MPI_Fint *count,
                          MPI_Fint *f_datatype, MPI_Fint *f_op, MPI_Fint *f_comm,
                          MPI_Fint *f_error) {

   MPI_Datatype c_datatype = PMPI_Type_f2c(*f_datatype);
   MPI_Op c_op = PMPI_Op_f2c(*f_op);
   MPI_Comm c_comm = PMPI_Comm_f2c(*f_comm);

   sendbuf = (void*) vftr_is_F_MPI_IN_PLACE(sendbuf) ? MPI_IN_PLACE : sendbuf;
   sendbuf = (void*) vftr_is_F_MPI_BOTTOM(sendbuf) ? MPI_BOTTOM : sendbuf;
   recvbuf = (void*) vftr_is_F_MPI_BOTTOM(recvbuf) ? MPI_BOTTOM : recvbuf;

   int c_error = vftr_MPI_Allreduce(sendbuf,
                                    recvbuf,
                                    (int)(*count),
                                    c_datatype,
                                    c_op,
                                    c_comm);

   *f_error = (MPI_Fint) (c_error);
}

void vftr_MPI_Alltoall_F(void *sendbuf, MPI_Fint *sendcount, MPI_Fint *f_sendtype,
                         void *recvbuf, MPI_Fint *recvcount, MPI_Fint *f_recvtype,
                         MPI_Fint *f_comm, MPI_Fint *f_error) {

   MPI_Datatype c_sendtype = PMPI_Type_f2c(*f_sendtype);
   MPI_Datatype c_recvtype = PMPI_Type_f2c(*f_recvtype);
   MPI_Comm c_comm = PMPI_Comm_f2c(*f_comm);

   sendbuf = (void*) vftr_is_F_MPI_IN_PLACE(sendbuf) ? MPI_IN_PLACE : sendbuf;
   sendbuf = (void*) vftr_is_F_MPI_BOTTOM(sendbuf) ? MPI_BOTTOM : sendbuf;
   recvbuf = (void*) vftr_is_F_MPI_BOTTOM(recvbuf) ? MPI_BOTTOM : recvbuf;

   int c_error = vftr_MPI_Alltoall(sendbuf,
                                   (int)(*sendcount),
                                   c_sendtype,
                                   recvbuf,
                                   (int)(*recvcount),
                                   c_recvtype,
                                   c_comm);

   *f_error = (MPI_Fint) (c_error);
}

void vftr_MPI_Alltoallv_F(void *sendbuf, MPI_Fint *f_sendcounts, MPI_Fint *f_sdispls, MPI_Fint *f_sendtype,
                          void *recvbuf, MPI_Fint *f_recvcounts, MPI_Fint *f_rdispls, MPI_Fint *f_recvtype,
                          MPI_Fint *f_comm, MPI_Fint *f_error) {

   MPI_Comm c_comm = PMPI_Comm_f2c(*f_comm);
   int size;
   PMPI_Comm_size(c_comm, &size);
   int *c_sendcounts = (int*) malloc(size*sizeof(int));
   for (int i=0; i<size; i++) {
      c_sendcounts[i] = (int) f_sendcounts[i];
   }
   int *c_sdispls = (int*) malloc(size*sizeof(int));
   for (int i=0; i<size; i++) {
      c_sdispls[i] = (int) f_sdispls[i];
   }
   MPI_Datatype c_sendtype = PMPI_Type_f2c(*f_sendtype);

   int *c_recvcounts = (int*) malloc(size*sizeof(int));
   for (int i=0; i<size; i++) {
      c_recvcounts[i] = (int) f_recvcounts[i];
   }
   int *c_rdispls = (int*) malloc(size*sizeof(int));
   for (int i=0; i<size; i++) {
      c_rdispls[i] = (int) f_rdispls[i];
   }
   MPI_Datatype c_recvtype = PMPI_Type_f2c(*f_recvtype);

   sendbuf = (void*) vftr_is_F_MPI_IN_PLACE(sendbuf) ? MPI_IN_PLACE : sendbuf;
   sendbuf = (void*) vftr_is_F_MPI_BOTTOM(sendbuf) ? MPI_BOTTOM : sendbuf;
   recvbuf = (void*) vftr_is_F_MPI_BOTTOM(recvbuf) ? MPI_BOTTOM : recvbuf;

   int c_error = vftr_MPI_Alltoallv(sendbuf,
                                    c_sendcounts,
                                    c_sdispls,
                                    c_sendtype,
                                    recvbuf,
                                    c_recvcounts,
                                    c_rdispls,
                                    c_recvtype,
                                    c_comm);
   free(c_sendcounts);
   free(c_sdispls);
   free(c_recvcounts);
   free(c_rdispls);

   *f_error = (MPI_Fint) (c_error);
}

void vftr_MPI_Alltoallw_F(void *sendbuf, MPI_Fint *f_sendcounts, MPI_Fint *f_sdispls, MPI_Fint *f_sendtypes,
                          void *recvbuf, MPI_Fint *f_recvcounts, MPI_Fint *f_rdispls, MPI_Fint *f_recvtypes,
                          MPI_Fint *f_comm, MPI_Fint *f_error) {

   MPI_Comm c_comm = PMPI_Comm_f2c(*f_comm);
   int size;
   PMPI_Comm_size(c_comm, &size);
   int *c_sendcounts = (int*) malloc(size*sizeof(int));
   for (int i=0; i<size; i++) {
      c_sendcounts[i] = (int) f_sendcounts[i];
   }
   int *c_sdispls = (int*) malloc(size*sizeof(int));
   for (int i=0; i<size; i++) {
      c_sdispls[i] = (int) f_sdispls[i];
   }
   MPI_Datatype *c_sendtypes = (MPI_Datatype*) malloc(size*sizeof(MPI_Datatype));
   for (int i=0; i<size; i++) {
      c_sendtypes[i] = PMPI_Type_f2c(f_sendtypes[i]);
   }

   int *c_recvcounts = (int*) malloc(size*sizeof(int));
   for (int i=0; i<size; i++) {
      c_recvcounts[i] = (int) f_recvcounts[i];
   }
   int *c_rdispls = (int*) malloc(size*sizeof(int));
   for (int i=0; i<size; i++) {
      c_rdispls[i] = (int) f_rdispls[i];
   }
   MPI_Datatype *c_recvtypes = (MPI_Datatype*) malloc(size*sizeof(MPI_Datatype));
   for (int i=0; i<size; i++) {
      c_recvtypes[i] = PMPI_Type_f2c(f_recvtypes[i]);
   }

   sendbuf = (void*) vftr_is_F_MPI_IN_PLACE(sendbuf) ? MPI_IN_PLACE : sendbuf;
   sendbuf = (void*) vftr_is_F_MPI_BOTTOM(sendbuf) ? MPI_BOTTOM : sendbuf;
   recvbuf = (void*) vftr_is_F_MPI_BOTTOM(recvbuf) ? MPI_BOTTOM : recvbuf;

   int c_error = vftr_MPI_Alltoallw(sendbuf,
                                    c_sendcounts,
                                    c_sdispls,
                                    c_sendtypes,
                                    recvbuf,
                                    c_recvcounts,
                                    c_rdispls,
                                    c_recvtypes,
                                    c_comm);

   free(c_sendcounts);
   free(c_sdispls);
   free(c_sendtypes);
   free(c_recvcounts);
   free(c_rdispls);
   free(c_recvtypes);

   *f_error = (MPI_Fint) (c_error);
}

void vftr_MPI_Bcast_F(void *buffer, MPI_Fint *count, MPI_Fint *f_datatype, 
                      MPI_Fint *root, MPI_Fint *f_comm, MPI_Fint *f_error) {

   MPI_Datatype c_datatype = PMPI_Type_f2c(*f_datatype);
   MPI_Comm c_comm = PMPI_Comm_f2c(*f_comm);

   int c_error = vftr_MPI_Bcast(buffer,
                                (int)(*count),
                                c_datatype,
                                (int)(*root),
                                c_comm);

   *f_error = (MPI_Fint) (c_error);
}

void vftr_MPI_Gather_F(void *sendbuf, MPI_Fint *sendcount, MPI_Fint *f_sendtype,
                       void *recvbuf, MPI_Fint *recvcount, MPI_Fint *f_recvtype,
                       MPI_Fint *root, MPI_Fint *f_comm, MPI_Fint *f_error) {

   MPI_Datatype c_sendtype = PMPI_Type_f2c(*f_sendtype);
   MPI_Datatype c_recvtype = PMPI_Type_f2c(*f_recvtype);
   MPI_Comm c_comm = PMPI_Comm_f2c(*f_comm);

   sendbuf = (void*) vftr_is_F_MPI_IN_PLACE(sendbuf) ? MPI_IN_PLACE : sendbuf;
   sendbuf = (void*) vftr_is_F_MPI_BOTTOM(sendbuf) ? MPI_BOTTOM : sendbuf;
   recvbuf = (void*) vftr_is_F_MPI_BOTTOM(recvbuf) ? MPI_BOTTOM : recvbuf;

   int c_error = vftr_MPI_Gather(sendbuf,
                                 (int)(*sendcount),
                                 c_sendtype,
                                 recvbuf,
                                 (int)(*recvcount),
                                 c_recvtype,
                                 (int)(*root),
                                 c_comm);

   *f_error = (MPI_Fint) (c_error);
}

void vftr_MPI_Gatherv_F(void *sendbuf, MPI_Fint *sendcount, MPI_Fint *f_sendtype,
                        void *recvbuf, MPI_Fint *f_recvcounts, MPI_Fint *f_displs,
                        MPI_Fint *f_recvtype, MPI_Fint *root, MPI_Fint *f_comm,
                        MPI_Fint *f_error) {

   MPI_Comm c_comm = PMPI_Comm_f2c(*f_comm);
   int size;
   PMPI_Comm_size(c_comm, &size);
   MPI_Datatype c_sendtype = PMPI_Type_f2c(*f_sendtype);
   int *c_recvcounts = (int*) malloc(size*sizeof(int));
   for (int i=0; i<size; i++) {
      c_recvcounts[i] = (int) f_recvcounts[i];
   }
   int *c_displs = (int*) malloc(size*sizeof(int));
   for (int i=0; i<size; i++) {
      c_displs[i] = (int) f_displs[i];
   }
   MPI_Datatype c_recvtype = PMPI_Type_f2c(*f_recvtype);

   sendbuf = (void*) vftr_is_F_MPI_IN_PLACE(sendbuf) ? MPI_IN_PLACE : sendbuf;
   sendbuf = (void*) vftr_is_F_MPI_BOTTOM(sendbuf) ? MPI_BOTTOM : sendbuf;
   recvbuf = (void*) vftr_is_F_MPI_BOTTOM(recvbuf) ? MPI_BOTTOM : recvbuf;

   int c_error = vftr_MPI_Gatherv(sendbuf,
                                  (int)(*sendcount),
                                  c_sendtype,
                                  recvbuf,
                                  c_recvcounts,
                                  c_displs,
                                  c_recvtype,
                                  (int)(*root),
                                  c_comm);

   free(c_recvcounts);
   free(c_displs);

   *f_error = (MPI_Fint) (c_error);
}

void vftr_MPI_Reduce_F(void *sendbuf, void *recvbuf, MPI_Fint *count, MPI_Fint *f_datatype,
                       MPI_Fint *f_op, MPI_Fint *root, MPI_Fint *f_comm, MPI_Fint *f_error) {

   MPI_Datatype c_datatype = PMPI_Type_f2c(*f_datatype);
   MPI_Op c_op = PMPI_Op_f2c(*f_op);
   MPI_Comm c_comm = PMPI_Comm_f2c(*f_comm);

   sendbuf = (void*) vftr_is_F_MPI_IN_PLACE(sendbuf) ? MPI_IN_PLACE : sendbuf;
   sendbuf = (void*) vftr_is_F_MPI_BOTTOM(sendbuf) ? MPI_BOTTOM : sendbuf;
   recvbuf = (void*) vftr_is_F_MPI_BOTTOM(recvbuf) ? MPI_BOTTOM : recvbuf;

   int c_error = vftr_MPI_Reduce(sendbuf,
                                 recvbuf,
                                 (int)(*count),
                                 c_datatype,
                                 c_op,
                                 (int)(*root),
                                 c_comm);

   *f_error = (MPI_Fint) (c_error);
}

void vftr_MPI_Reduce_scatter_F(void *sendbuf, void *recvbuf, MPI_Fint *f_recvcounts,
                               MPI_Fint *f_datatype, MPI_Fint *f_op, MPI_Fint *f_comm,
                               MPI_Fint *f_error) {

   MPI_Comm c_comm = PMPI_Comm_f2c(*f_comm);
   int size;
   PMPI_Comm_size(c_comm, &size);
   int *c_recvcounts = (int*) malloc(size*sizeof(int));
   for (int i=0; i<size; i++) {
      c_recvcounts[i] = (int) f_recvcounts[i];
   }
   MPI_Datatype c_datatype = PMPI_Type_f2c(*f_datatype);
   MPI_Op c_op = PMPI_Op_f2c(*f_op);

   sendbuf = (void*) vftr_is_F_MPI_IN_PLACE(sendbuf) ? MPI_IN_PLACE : sendbuf;
   sendbuf = (void*) vftr_is_F_MPI_BOTTOM(sendbuf) ? MPI_BOTTOM : sendbuf;
   recvbuf = (void*) vftr_is_F_MPI_BOTTOM(recvbuf) ? MPI_BOTTOM : recvbuf;

   int c_error = vftr_MPI_Reduce_scatter(sendbuf,
                                         recvbuf,
                                         c_recvcounts,
                                         c_datatype,
                                         c_op,
                                         c_comm);

   free(c_recvcounts);

   *f_error = (MPI_Fint) (c_error);
}

void vftr_MPI_Scatter_F(void *sendbuf, MPI_Fint *sendcount, MPI_Fint *f_sendtype,
                        void *recvbuf, MPI_Fint *recvcount, MPI_Fint *f_recvtype,
                        MPI_Fint *root, MPI_Fint *f_comm, MPI_Fint *f_error) {

   MPI_Datatype c_sendtype = PMPI_Type_f2c(*f_sendtype);
   MPI_Datatype c_recvtype = PMPI_Type_f2c(*f_recvtype);
   MPI_Comm c_comm = PMPI_Comm_f2c(*f_comm);

   sendbuf = (void*) vftr_is_F_MPI_BOTTOM(sendbuf) ? MPI_BOTTOM : sendbuf;
   recvbuf = (void*) vftr_is_F_MPI_IN_PLACE(recvbuf) ? MPI_IN_PLACE : recvbuf;
   recvbuf = (void*) vftr_is_F_MPI_BOTTOM(recvbuf) ? MPI_BOTTOM : recvbuf;

   int c_error = vftr_MPI_Scatter(sendbuf,
                                  (int)(*sendcount),
                                  c_sendtype,
                                  recvbuf,
                                  (int)(*recvcount),
                                  c_recvtype,
                                  (int)(*root),
                                  c_comm);

   *f_error = (MPI_Fint) (c_error);
}

void vftr_MPI_Scatterv_F(void *sendbuf, MPI_Fint *f_sendcounts, MPI_Fint *f_displs,
                         MPI_Fint *f_sendtype, void *recvbuf, MPI_Fint *recvcount,
                         MPI_Fint *f_recvtype, MPI_Fint *root, MPI_Fint *f_comm,
                         MPI_Fint *f_error) {


   MPI_Comm c_comm = PMPI_Comm_f2c(*f_comm);
   int size;
   PMPI_Comm_size(c_comm, &size);
   int *c_sendcounts = (int*) malloc(size*sizeof(int));
   for (int i=0; i<size; i++) {
      c_sendcounts[i] = (int) f_sendcounts[i];
   }
   int *c_displs = (int*) malloc(size*sizeof(int));
   for (int i=0; i<size; i++) {
      c_displs[i] = (int) f_displs[i];
   }
   MPI_Datatype c_sendtype = PMPI_Type_f2c(*f_sendtype);
   MPI_Datatype c_recvtype = PMPI_Type_f2c(*f_recvtype);

   sendbuf = (void*) vftr_is_F_MPI_BOTTOM(sendbuf) ? MPI_BOTTOM : sendbuf;
   recvbuf = (void*) vftr_is_F_MPI_IN_PLACE(recvbuf) ? MPI_IN_PLACE : recvbuf;
   recvbuf = (void*) vftr_is_F_MPI_BOTTOM(recvbuf) ? MPI_BOTTOM : recvbuf;

   int c_error = vftr_MPI_Scatterv(sendbuf,
                                   c_sendcounts,
                                   c_displs,
                                   c_sendtype,
                                   recvbuf,
                                   (int)(*recvcount),
                                   c_recvtype,
                                   (int)(*root),
                                   c_comm);

   free(c_sendcounts);
   free(c_displs);

   *f_error = (MPI_Fint) (c_error);
}

#endif
