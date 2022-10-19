#ifdef _MPI
#include <mpi.h>

#include "mpi_buf_addr_const.h"
#include "neighbor_alltoallw.h"

int vftr_MPI_Neighbor_alltoallw_c2vftr(const void *sendbuf, const int *sendcounts,
                                       const MPI_Aint *sdispls, const MPI_Datatype *sendtypes,
                                       void *recvbuf, const int *recvcounts,
                                       const MPI_Aint *rdispls, const MPI_Datatype *recvtypes,
                                       MPI_Comm comm) {
   // determine the topology of the communicator
   int topology;
   PMPI_Topo_test(comm, &topology);
   switch(topology) {
      case MPI_GRAPH:
         return vftr_MPI_Neighbor_alltoallw_graph(sendbuf, sendcounts, sdispls, sendtypes,
                                                  recvbuf, recvcounts, rdispls, recvtypes,
                                                  comm);
         break;
      case MPI_CART:
         return vftr_MPI_Neighbor_alltoallw_cart(sendbuf, sendcounts, sdispls, sendtypes,
                                                 recvbuf, recvcounts, rdispls, recvtypes,
                                                 comm);
         break;
      case MPI_DIST_GRAPH:
         return vftr_MPI_Neighbor_alltoallw_dist_graph(sendbuf, sendcounts, sdispls, sendtypes,
                                                       recvbuf, recvcounts, rdispls, recvtypes,
                                                       comm);
         break;
      case MPI_UNDEFINED:
      default:
         return PMPI_Neighbor_alltoallw(sendbuf, sendcounts, sdispls, sendtypes,
                                        recvbuf, recvcounts, rdispls, recvtypes,
                                        comm);
   }
}

#endif
