#ifdef _MPI
#include <mpi.h>

#include "neighbor_allgather.h"

void vftr_MPI_Neighbor_allgather_f2vftr(void *sendbuf, MPI_Fint *sendcount,
                                        MPI_Fint *f_sendtype, void *recvbuf,
                                        MPI_Fint *recvcount, MPI_Fint *f_recvtype,
                                        MPI_Fint *f_comm, MPI_Fint *f_error) {

   MPI_Datatype c_sendtype = PMPI_Type_f2c(*f_sendtype);
   MPI_Datatype c_recvtype = PMPI_Type_f2c(*f_recvtype);
   MPI_Comm c_comm = PMPI_Comm_f2c(*f_comm);

   int c_error;
   // determine the topology of the communicator
   int topology;
   PMPI_Topo_test(c_comm, &topology);
   switch(topology) {
      case MPI_GRAPH:
         c_error = vftr_MPI_Neighbor_allgather_graph(sendbuf,
                                                     (int)(*sendcount),
                                                     c_sendtype,
                                                     recvbuf,
                                                     (int)(*recvcount),
                                                     c_recvtype,
                                                     c_comm);
         break;
      case MPI_CART:
         c_error = vftr_MPI_Neighbor_allgather_cart(sendbuf,
                                                    (int)(*sendcount),
                                                    c_sendtype,
                                                    recvbuf,
                                                    (int)(*recvcount),
                                                    c_recvtype,
                                                    c_comm);
         break;
      case MPI_DIST_GRAPH:
         c_error = vftr_MPI_Neighbor_allgather_dist_graph(sendbuf,
                                                          (int)(*sendcount),
                                                          c_sendtype,
                                                          recvbuf,
                                                          (int)(*recvcount),
                                                          c_recvtype,
                                                          c_comm);
         break;
      case MPI_UNDEFINED:
      default:
         c_error = PMPI_Neighbor_allgather(sendbuf,
                                           (int)(*sendcount),
                                           c_sendtype,
                                           recvbuf,
                                           (int)(*recvcount),
                                           c_recvtype,
                                           c_comm);
   }

   *f_error = (MPI_Fint) (c_error);
}

#endif
