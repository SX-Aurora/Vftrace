#ifdef _MPI
#include <mpi.h>

#include "mpi_buf_addr_const.h"
#include "neighbor_alltoallv.h"

int vftr_MPI_Neighbor_alltoallv_c2vftr(const void *sendbuf, const int *sendcounts,
                                       const int *sdispls, MPI_Datatype sendtype,
                                       void *recvbuf, const int *recvcounts,
                                       const int *rdispls, MPI_Datatype recvtype,
                                       MPI_Comm comm) {
   // determine the topology of the communicator
   int topology;
   PMPI_Topo_test(comm, &topology);
   switch(topology) {
      case MPI_GRAPH:
         return vftr_MPI_Neighbor_alltoallv_graph(sendbuf, sendcounts, sdispls, sendtype,
                                                  recvbuf, recvcounts, rdispls, recvtype,
                                                  comm);
         break;
      case MPI_CART:
         return vftr_MPI_Neighbor_alltoallv_cart(sendbuf, sendcounts, sdispls, sendtype,
                                                 recvbuf, recvcounts, rdispls, recvtype,
                                                 comm);
         break;
      case MPI_DIST_GRAPH:
         return vftr_MPI_Neighbor_alltoallv_dist_graph(sendbuf, sendcounts, sdispls, sendtype,
                                                       recvbuf, recvcounts, rdispls, recvtype,
                                                       comm);
         break;
      case MPI_UNDEFINED:
      default:
         return PMPI_Neighbor_alltoallv(sendbuf, sendcounts, sdispls, sendtype,
                                        recvbuf, recvcounts, rdispls, recvtype,
                                        comm);
   }
}

#endif
