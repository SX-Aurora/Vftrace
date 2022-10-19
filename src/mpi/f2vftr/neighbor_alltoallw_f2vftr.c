#ifdef _MPI
#include <mpi.h>

#include <stdlib.h>

#include "neighbor_alltoallw.h"

void vftr_MPI_Neighbor_alltoallw_f2vftr(void *sendbuf, MPI_Fint *f_sendcounts,
                                        MPI_Aint *f_sdispls, MPI_Fint *f_sendtypes,
                                        void *recvbuf, MPI_Fint *f_recvcounts,
                                        MPI_Aint *f_rdispls, MPI_Fint *f_recvtypes,
                                        MPI_Fint *f_comm, MPI_Fint *f_error) {

   MPI_Comm c_comm = PMPI_Comm_f2c(*f_comm);

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
   MPI_Aint *c_sdispls = (MPI_Aint*) malloc(sizeout*sizeof(MPI_Aint));
   for (int i=0; i<sizeout; i++) {
      c_sdispls[i] = (MPI_Aint) f_sdispls[i];
   }
   MPI_Datatype *c_sendtypes = (MPI_Datatype*) malloc(sizeout*sizeof(MPI_Datatype));
   for (int i=0; i<sizeout; i++) {
      c_sendtypes[i] = PMPI_Type_f2c(f_sendtypes[i]);
   }

   int *c_recvcounts = (int*) malloc(sizein*sizeof(int));
   for (int i=0; i<sizein; i++) {
      c_recvcounts[i] = (int) f_recvcounts[i];
   }
   MPI_Aint *c_rdispls = (MPI_Aint*) malloc(sizein*sizeof(MPI_Aint));
   for (int i=0; i<sizein; i++) {
      c_rdispls[i] = (MPI_Aint) f_rdispls[i];
   }
   MPI_Datatype *c_recvtypes = (MPI_Datatype*) malloc(sizein*sizeof(MPI_Datatype));
   for (int i=0; i<sizein; i++) {
      c_recvtypes[i] = PMPI_Type_f2c(f_recvtypes[i]);
   }

   int c_error;
   switch(topology) {
      case MPI_GRAPH:
         c_error = vftr_MPI_Neighbor_alltoallw_graph(sendbuf,
                                                     c_sendcounts,
                                                     c_sdispls,
                                                     c_sendtypes,
                                                     recvbuf,
                                                     c_recvcounts,
                                                     c_rdispls,
                                                     c_recvtypes,
                                                     c_comm);
         break;
      case MPI_CART:
         c_error = vftr_MPI_Neighbor_alltoallw_cart(sendbuf,
                                                    c_sendcounts,
                                                    c_sdispls,
                                                    c_sendtypes,
                                                    recvbuf,
                                                    c_recvcounts,
                                                    c_rdispls,
                                                    c_recvtypes,
                                                    c_comm);
         break;
      case MPI_DIST_GRAPH:
         c_error = vftr_MPI_Neighbor_alltoallw_dist_graph(sendbuf,
                                                          c_sendcounts,
                                                          c_sdispls,
                                                          c_sendtypes,
                                                          recvbuf,
                                                          c_recvcounts,
                                                          c_rdispls,
                                                          c_recvtypes,
                                                          c_comm);
         break;
   }

   free(c_sendcounts);
   free(c_sdispls);
   free(c_sendtypes);
   free(c_recvcounts);
   free(c_rdispls);
   free(c_recvtypes);

   *f_error = (MPI_Fint) (c_error);
}

#endif
