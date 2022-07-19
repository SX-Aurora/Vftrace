#include <stdlib.h>
#include <stdio.h>

#ifdef _MPI
#include <mpi.h>
#endif

int main(int argc, char** argv) {
#ifdef _MPI
   MPI_Init(&argc, &argv);

   // Get number or processes
   int comm_size;
   MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
   // Get rank of process
   int my_rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

   // send/recv buffer
   int nints = 1<<my_rank;
   int *scounts = (int*) malloc(comm_size*sizeof(int));
   int *sdispls = (int*) malloc(comm_size*sizeof(int));
   int nstot = 0;
   for (int i=0; i<comm_size; i++) {
      scounts[i] = nints;
      sdispls[i] = nstot;
      nstot += scounts[i];
   }
   int *sbuffer = (int*) malloc(nstot*sizeof(int));
   for (int i=0; i<nstot; i++) {sbuffer[i]=my_rank;}

   // prepare special arrays for recv
   int *rcounts = (int*) malloc(comm_size*sizeof(int));
   int *rdispls = (int*) malloc(comm_size*sizeof(int));
   int nrtot = 0;
   for (int i=0; i<comm_size; i++) {
      rcounts[i] = 1<<i;
      rdispls[i] = nrtot;
      nrtot += rcounts[i];
   }
   int *rbuffer = (int*) malloc(nrtot*sizeof(int));
   for (int i=0; i<nrtot; i++) {rbuffer[i] = -1;}

   // Messaging
   MPI_Alltoallv(sbuffer, scounts, sdispls, MPI_INT,
                 rbuffer, rcounts, rdispls, MPI_INT,
                 MPI_COMM_WORLD);

   for (int irank=0; irank<comm_size; irank++) {
      if (my_rank == irank) {
         printf("%1d: scounts =", my_rank);
         for (int i=0; i<comm_size; i++) {
            printf(" %2d", scounts[i]);
         } printf("\n");
         printf("%1d: sdispls =", my_rank);
         for (int i=0; i<comm_size; i++) {
            printf(" %2d", sdispls[i]);
         } printf("\n");
         printf("%1d: sbuffer =", my_rank);
         for (int i=0; i<nstot; i++) {
            printf(" %2d", sbuffer[i]);
         } printf("\n");
         printf("%1d: rcounts =", my_rank);
         for (int i=0; i<comm_size; i++) {
            printf(" %2d", rcounts[i]);
         } printf("\n");
         printf("%1d: rdispls =", my_rank);
         for (int i=0; i<comm_size; i++) {
            printf(" %2d", rdispls[i]);
         } printf("\n");
         printf("%1d: rbuffer =", my_rank);
         for (int i=0; i<nrtot; i++) {
            printf(" %2d", rbuffer[i]);
         } printf("\n");
         printf("\n");
      }
      fflush(stdout);
      MPI_Barrier(MPI_COMM_WORLD);
   }

   free(sbuffer);
   sbuffer=NULL;

   free(scounts);
   scounts=NULL;

   free(sdispls);
   sdispls=NULL;

   free(rbuffer);
   rbuffer=NULL;

   free(rcounts);
   rcounts=NULL;

   free(rdispls);
   rdispls=NULL;

   MPI_Finalize();
#endif

   return 0;
}
