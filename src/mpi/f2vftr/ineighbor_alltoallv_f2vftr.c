#ifdef _MPI
#include <mpi.h>

#include <stdlib.h>

#include "ineighbor_alltoallv.h"

void vftr_MPI_Ineighbor_alltoallv_f2vftr(void *sendbuf, MPI_Fint *f_sendcounts,
                                         MPI_Fint *f_sdispls, MPI_Fint *f_sendtype,
                                         void *recvbuf, MPI_Fint *f_recvcounts,
                                         MPI_Fint *f_rdispls, MPI_Fint *f_recvtype,
                                         MPI_Fint *f_comm, MPI_Fint *f_request,
                                         MPI_Fint *f_error) {

   MPI_Comm c_comm = PMPI_Comm_f2c(*f_comm);
   MPI_Request c_request;

   int sizein;
   int sizeout;
   // determine the topology of the communicator
   int topology;
   PMPI_Topo_test(c_comm, &topology);
   switch(topology) {
      case MPI_GRAPH:;
         int rank;
         PMPI_Comm_rank(c_comm, &rank);
         PMPI_Graph_neighbors_count(c_comm, rank, &sizein);
         sizeout = sizein;
         break;
      case MPI_CART:
         PMPI_Cartdim_get(c_comm, &sizein);
         // Number of neighbors for cartesian communicators is always 2*ndims
         sizein *= 2;
         sizeout = sizein;
         break;
      case MPI_DIST_GRAPH:;
         int weighted;
         PMPI_Dist_graph_neighbors_count(c_comm, &sizein,
                                         &sizeout, &weighted);
         break;
   }

   int *c_sendcounts = (int*) malloc(sizeout*sizeof(int));
   for (int i=0; i<sizeout; i++) {
      c_sendcounts[i] = (int) f_sendcounts[i];
   }
   int *c_sdispls = (int*) malloc(sizeout*sizeof(int));
   for (int i=0; i<sizeout; i++) {
      c_sdispls[i] = (int) f_sdispls[i];
   }

   int *c_recvcounts = (int*) malloc(sizein*sizeof(int));
   for (int i=0; i<sizein; i++) {
      c_recvcounts[i] = (int) f_recvcounts[i];
   }
   int *c_rdispls = (int*) malloc(sizein*sizeof(int));
   for (int i=0; i<sizein; i++) {
      c_rdispls[i] = (int) f_rdispls[i];
   }

   MPI_Datatype c_sendtype = PMPI_Type_f2c(*f_sendtype);
   MPI_Datatype c_recvtype = PMPI_Type_f2c(*f_recvtype);

   int c_error;
   switch(topology) {
      case MPI_GRAPH:
         c_error = vftr_MPI_Ineighbor_alltoallv_graph(sendbuf,
                                                      c_sendcounts,
                                                      c_sdispls,
                                                      c_sendtype,
                                                      recvbuf,
                                                      c_recvcounts,
                                                      c_rdispls,
                                                      c_recvtype,
                                                      c_comm,
                                                      &c_request);
         break;
      case MPI_CART:
         c_error = vftr_MPI_Ineighbor_alltoallv_cart(sendbuf,
                                                     c_sendcounts,
                                                     c_sdispls,
                                                     c_sendtype,
                                                     recvbuf,
                                                     c_recvcounts,
                                                     c_rdispls,
                                                     c_recvtype,
                                                     c_comm,
                                                     &c_request);
         break;
      case MPI_DIST_GRAPH:
         c_error = vftr_MPI_Ineighbor_alltoallv_dist_graph(sendbuf,
                                                           c_sendcounts,
                                                           c_sdispls,
                                                           c_sendtype,
                                                           recvbuf,
                                                           c_recvcounts,
                                                           c_rdispls,
                                                           c_recvtype,
                                                           c_comm,
                                                           &c_request);
         break;
   }

   *f_error = (MPI_Fint) (c_error);
   *f_request = PMPI_Request_c2f(c_request);
}

#endif
