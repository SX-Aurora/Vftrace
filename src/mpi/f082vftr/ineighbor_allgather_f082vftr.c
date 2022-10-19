#ifdef _MPI
#include <mpi.h>

#include "ineighbor_allgather.h"

void vftr_MPI_Ineighbor_allgather_f082vftr(void *sendbuf, MPI_Fint *sendcount,
                                           MPI_Fint *f_sendtype, void *recvbuf,
                                           MPI_Fint *recvcount, MPI_Fint *f_recvtype,
                                           MPI_Fint *f_comm, MPI_Fint *f_request,
                                           MPI_Fint *f_error) {

   MPI_Datatype c_sendtype = PMPI_Type_f2c(*f_sendtype);
   MPI_Datatype c_recvtype = PMPI_Type_f2c(*f_recvtype);
   MPI_Comm c_comm = PMPI_Comm_f2c(*f_comm);
   MPI_Request c_request;

   int c_error;
   // determine the topology of the communicator
   int topology;
   PMPI_Topo_test(c_comm, &topology);
   switch(topology) {
      case MPI_GRAPH:
         c_error = vftr_MPI_Ineighbor_allgather_graph(sendbuf,
                                                      (int)(*sendcount),
                                                      c_sendtype,
                                                      recvbuf,
                                                      (int)(*recvcount),
                                                      c_recvtype,
                                                      c_comm,
                                                      &c_request);
         break;
      case MPI_CART:
         c_error = vftr_MPI_Ineighbor_allgather_cart(sendbuf,
                                                     (int)(*sendcount),
                                                     c_sendtype,
                                                     recvbuf,
                                                     (int)(*recvcount),
                                                     c_recvtype,
                                                     c_comm,
                                                     &c_request);
         break;
      case MPI_DIST_GRAPH:
         c_error = vftr_MPI_Ineighbor_allgather_dist_graph(sendbuf,
                                                           (int)(*sendcount),
                                                           c_sendtype,
                                                           recvbuf,
                                                           (int)(*recvcount),
                                                           c_recvtype,
                                                           c_comm,
                                                           &c_request);
         break;
      case MPI_UNDEFINED:
      default:
         c_error = PMPI_Ineighbor_allgather(sendbuf,
                                            (int)(*sendcount),
                                            c_sendtype,
                                            recvbuf,
                                            (int)(*recvcount),
                                            c_recvtype,
                                            c_comm,
                                            &c_request);
   }

   *f_error = (MPI_Fint) (c_error);
   *f_request = PMPI_Request_c2f(c_request);
}

#endif
