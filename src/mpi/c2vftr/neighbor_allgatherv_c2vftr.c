#ifdef _MPI
#include <mpi.h>

#include "mpi_buf_addr_const.h"
#include "neighbor_allgatherv.h"

int vftr_MPI_Neighbor_allgatherv_c2vftr(const void *sendbuf, int sendcount,
                                        MPI_Datatype sendtype, void *recvbuf,
                                        const int *recvcounts, const int *displs,
                                        MPI_Datatype recvtype, MPI_Comm comm) {
   // determine the topology of the communicator
   int topology;
   PMPI_Topo_test(comm, &topology);
   switch(topology) {
      case MPI_GRAPH:
         return vftr_MPI_Neighbor_allgatherv_graph(sendbuf, sendcount, sendtype,
                                                   recvbuf, recvcounts, displs, recvtype,
                                                   comm);
         break;
      case MPI_CART:
         return vftr_MPI_Neighbor_allgatherv_cart(sendbuf, sendcount, sendtype,
                                                  recvbuf, recvcounts, displs, recvtype,
                                                  comm);
         break;
      case MPI_DIST_GRAPH:
         return vftr_MPI_Neighbor_allgatherv_dist_graph(sendbuf, sendcount, sendtype,
                                                        recvbuf, recvcounts, displs, recvtype,
                                                        comm);
         break;
      case MPI_UNDEFINED:
      default:
         return PMPI_Neighbor_allgatherv(sendbuf, sendcount, sendtype,
                                        recvbuf, recvcounts, displs, recvtype,
                                        comm);
   }
}

#endif
