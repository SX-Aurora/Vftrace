#include <stdlib.h>

#include <mpi.h>

#include "mpi_buf_addr_const.h"
#include "ineighbor_allgatherv.h"

int vftr_MPI_Ineighbor_allgatherv_c2vftr(const void *sendbuf, int sendcount,
                                         MPI_Datatype sendtype, void *recvbuf,
                                         const int *recvcounts, const int *displs,
                                         MPI_Datatype recvtype, MPI_Comm comm,
                                         MPI_Request *request) {
   // create a copy of recvcount and displs
   // It will be deallocated upon completion of the request
   int sizein;
   // determine the topology of the communicator
   int topology;
   PMPI_Topo_test(comm, &topology);
   switch(topology) {
      case MPI_GRAPH:;
         int rank;
         PMPI_Comm_rank(comm, &rank);
         PMPI_Graph_neighbors_count(comm, rank, &sizein);
         break;
      case MPI_CART:
         PMPI_Cartdim_get(comm, &sizein);
         // Number of neighbors for cartesian communicators is always 2*ndims
         sizein *= 2;
         break;
      case MPI_DIST_GRAPH:;
         int sizeout;
         int weighted;
         PMPI_Dist_graph_neighbors_count(comm, &sizein,
                                         &sizeout, &weighted);
         break;
   }

   int *tmp_recvcounts = (int*) malloc(sizein*sizeof(int));
   for (int i=0; i<sizein; i++) {
      tmp_recvcounts[i] = recvcounts[i];
   }
   int *tmp_displs = (int*) malloc(sizein*sizeof(int));
   for (int i=0; i<sizein; i++) {
      tmp_displs[i] = displs[i];
   }

   switch(topology) {
      case MPI_GRAPH:
         return vftr_MPI_Ineighbor_allgatherv_graph(sendbuf, sendcount, sendtype,
                                                    recvbuf, tmp_recvcounts,
                                                    tmp_displs, recvtype,
                                                    comm, request);
         break;
      case MPI_CART:
         return vftr_MPI_Ineighbor_allgatherv_cart(sendbuf, sendcount, sendtype,
                                                   recvbuf, tmp_recvcounts,
                                                   tmp_displs, recvtype,
                                                   comm, request);
         break;
      case MPI_DIST_GRAPH:
         return vftr_MPI_Ineighbor_allgatherv_dist_graph(sendbuf, sendcount, sendtype,
                                                         recvbuf, tmp_recvcounts,
                                                         tmp_displs, recvtype,
                                                         comm, request);
         break;
      case MPI_UNDEFINED:
      default:
         // should never get here.
         // But if so, free the arrays
         free(tmp_recvcounts);
         free(tmp_displs);
         return PMPI_Ineighbor_allgatherv(sendbuf, sendcount, sendtype,
                                          recvbuf, recvcounts, displs,
                                          recvtype, comm,
                                          request);
   }
}
