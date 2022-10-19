#ifdef _MPI
#include <mpi.h>

#include "mpi_buf_addr_const.h"
#include "neighbor_alltoall.h"

int vftr_MPI_Neighbor_alltoall_c2vftr(const void *sendbuf, int sendcount,
                                      MPI_Datatype sendtype, void *recvbuf,
                                      int recvcount, MPI_Datatype recvtype,
                                      MPI_Comm comm) {
   // determine the topology of the communicator
   int topology;
   PMPI_Topo_test(comm, &topology);
   switch(topology) {
      case MPI_GRAPH:
         return vftr_MPI_Neighbor_alltoall_graph(sendbuf, sendcount, sendtype,
                                                 recvbuf, recvcount, recvtype,
                                                 comm);
         break;
      case MPI_CART:
         return vftr_MPI_Neighbor_alltoall_cart(sendbuf, sendcount, sendtype,
                                                recvbuf, recvcount, recvtype,
                                                comm);
         break;
      case MPI_DIST_GRAPH:
         return vftr_MPI_Neighbor_alltoall_dist_graph(sendbuf, sendcount, sendtype,
                                                      recvbuf, recvcount, recvtype,
                                                      comm);
         break;
      case MPI_UNDEFINED:
      default:
         return PMPI_Neighbor_alltoall(sendbuf, sendcount, sendtype,
                                       recvbuf, recvcount, recvtype,
                                       comm);
   }
}

#endif
