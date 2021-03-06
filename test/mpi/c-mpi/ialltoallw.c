#include <mpi.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char** argv) {
   MPI_Init(&argc, &argv);

   // Get number or processes
   int comm_size;
   MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
   // Get rank of process
   int my_rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

   // require cmd-line argument
   if (argc < 2) {
      printf("./ialltoallw <msgsize in ints>\n");
      return 1;
   }

   // allocating send/recv buffer
   int nints = atoi(argv[1]) + my_rank;

   // prepare special arrays for send
   int *scounts = (int*) malloc(comm_size*sizeof(int));
   int *sdispls = (int*) malloc(comm_size*sizeof(int));
   MPI_Datatype *stypes = (MPI_Datatype*) malloc(comm_size*sizeof(MPI_Datatype));
   int nstot = 0;
   for (int i=0; i<comm_size; i++) {
      scounts[i] = nints;
      sdispls[i] = nstot*sizeof(int);
      stypes[i] = MPI_INT;
      nstot += scounts[i];
   }
   int *sbuffer = (int*) malloc(nstot*sizeof(int));
   for (int i=0; i<nstot; i++) {sbuffer[i]=my_rank;}

   // prepare special arrays for recv
   int *rcounts = (int*) malloc(comm_size*sizeof(int));
   int *rdispls = (int*) malloc(comm_size*sizeof(int));
   MPI_Datatype *rtypes = (MPI_Datatype*) malloc(comm_size*sizeof(MPI_Datatype));
   int nrtot = 0;
   for (int i=0; i<comm_size; i++) {
      rcounts[i] = nints - my_rank + i;
      rdispls[i] = nrtot*sizeof(int);
      rtypes[i] = MPI_INT;
      nrtot += rcounts[i];
   }
   int *rbuffer = (int*) malloc(nrtot*sizeof(int));
   for (int i=0; i<nrtot; i++) {rbuffer[i] = -1;}

   // Messaging
   MPI_Request myrequest;
   MPI_Ialltoallw(sbuffer, scounts, sdispls, stypes,
                  rbuffer, rcounts, rdispls, rtypes,
                  MPI_COMM_WORLD, &myrequest);
   printf("Communicating with all ranks\n");
   MPI_Status mystatus;
   MPI_Wait(&myrequest, &mystatus);

   // validate data
   bool valid_data = true;
   for (int irank=0; irank<comm_size; irank++) {
      for (int i=0; i<rcounts[irank]; i++) {
         if (rbuffer[i+rdispls[irank]/sizeof(int)] != irank) {
            printf("Rank %d received faulty data from rank %d\n", my_rank, irank);
            valid_data = false;
            break;
         }
      }
   }

   free(sbuffer);
   sbuffer=NULL;

   free(scounts);
   scounts=NULL;

   free(sdispls);
   sdispls=NULL;

   free(stypes);
   stypes=NULL;

   free(rbuffer);
   rbuffer=NULL;

   free(rcounts);
   rcounts=NULL;

   free(rdispls);
   rdispls=NULL;

   free(rtypes);
   rtypes=NULL;

   MPI_Finalize();

   return valid_data ? 0 : 1;
}
