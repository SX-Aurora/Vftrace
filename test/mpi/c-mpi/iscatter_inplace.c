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
      printf("./iscatter_inplace <msgsize in ints>\n");
      return 1;
   }

   int rootrank = 0;
   // allocating send/recv buffer
   int nints = atoi(argv[1]);
   int *rbuffer = NULL;
   int recvcount;
   MPI_Datatype recvtype;
   int *sbuffer = NULL;
   if (my_rank == rootrank) {
      sbuffer = (int*) malloc(comm_size*nints*sizeof(int));
      for (int irank=0; irank<comm_size; irank++) {
         for (int i=0; i<nints; i++) {
            sbuffer[i+irank*nints] = irank;
         }
      }
      rbuffer = MPI_IN_PLACE;
      recvcount = 0;
      recvtype = MPI_DATATYPE_NULL;
   } else {
      rbuffer = (int*) malloc(nints*sizeof(int));
      for (int i=0; i<nints; i++) {rbuffer[i]=-1;}
      recvcount = nints;
      recvtype = MPI_INT;
   }

   // Messaging
   MPI_Request myrequest;
   MPI_Iscatter(sbuffer, nints, MPI_INT,
                rbuffer, recvcount, recvtype,
                rootrank, MPI_COMM_WORLD, &myrequest);
   if (my_rank == rootrank) {
      printf("Scattering messages from rank %d\n", my_rank);
   }
   MPI_Status mystatus;
   MPI_Wait(&myrequest, &mystatus);

   // validate data
   bool valid_data = true;
   if (my_rank == rootrank) {
      for (int i=0; i<nints; i++) {
         if (sbuffer[i+my_rank*nints] != my_rank) {
            printf("Rank %d received faulty data from rank %d\n", my_rank, rootrank);
            valid_data = false;
            break;
         }
      }
   } else {
      for (int i=0; i<nints; i++) {
         if (rbuffer[i] != my_rank) {
            printf("Rank %d received faulty data from rank %d\n", my_rank, rootrank);
            valid_data = false;
            break;
         }
      }
   }

   if (my_rank == rootrank) {
      free(sbuffer);
      sbuffer=NULL;
   } else {
      free(rbuffer);
      rbuffer=NULL;
   }

   MPI_Finalize();

   return valid_data ? 0 : 1;
}
