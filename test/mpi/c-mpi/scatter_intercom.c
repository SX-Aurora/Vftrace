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
   // require at least two processes
   if (comm_size < 2) {
      printf("At least two ranks are required\n");
      printf("Run again with '-np 2'\n");
      MPI_Finalize();
      return 1;
   }

   // require cmd-line argument
   if (argc < 2) {
      printf("./gather_intercom <msgsize in ints>\n");
      return 1;
   }

   // create intercommunicator
   int color = (2*my_rank) / comm_size;
   MPI_Comm sub_comm;
   MPI_Comm_split(MPI_COMM_WORLD,
                  color, my_rank, &sub_comm);
   // get local comm size and rank
   int sub_comm_size;
   MPI_Comm_size(sub_comm, &sub_comm_size);
   int my_sub_rank;
   MPI_Comm_rank(sub_comm, &my_sub_rank);

   MPI_Comm int_comm;
   int local_leader = 0;
   int remote_leader = (1-color)*(comm_size+1)/2;
   MPI_Intercomm_create(sub_comm, 
                        local_leader,
                        MPI_COMM_WORLD,
                        remote_leader, 1,
                        &int_comm);
   int sub_comm_remote_size;
   MPI_Comm_remote_size(int_comm, &sub_comm_remote_size);

   int rootrank = 0;
   // allocating send/recv buffer
   int nints = atoi(argv[1]);
   
   int *rbuffer = (int*) malloc(nints*sizeof(int));
   for (int i=0; i<nints; i++) {rbuffer[i]=-1;}
   int *sbuffer = NULL;
   if (my_rank == 0) {
      sbuffer = (int*) malloc(sub_comm_remote_size*nints*sizeof(int));
      for (int irank=0; irank<sub_comm_remote_size; irank++) {
         int jrank = sub_comm_size + irank;
         for (int i=0; i<nints; i++) {
            sbuffer[i+irank*nints] = jrank;
         }
      }
   }

   // preparing the intercomm root assignment
   int root;
   if (color == 0) {
      // Sub communicator of receiving group
      if (my_sub_rank == 0) {
         // Receiving rank in receiving subgroup
         root = MPI_ROOT;
      } else {
         // Ideling processes
         root = MPI_PROC_NULL;
      }
   } else {
      // Sub communicator or sending group
      root = 0; // receiving rank in remote group
   }
   MPI_Scatter(sbuffer, nints, MPI_INT,
               rbuffer, nints, MPI_INT,
               root, int_comm);
   if (my_rank == 0) {
      printf("Scattering from global rank %d\n", my_rank);
      printf("(Group=%d, local rank=%d)\n", color, my_sub_rank);
   }

   // validate data
   bool valid_data = true;
   if (color == 0) {
      for (int i=0; i<nints; i++){
         if (rbuffer[i] != -1) {
            printf("Rank %d has no longer valid data\n", my_rank);
            valid_data = false;
            break;
         }
      }
   } else {
      for (int i=0; i<nints; i++){
         if (rbuffer[i] != my_rank) {
            printf("Rank %d received faulty data from rank %d\n", my_rank, rootrank);
            valid_data = false;
            break;
         }
      }
   }
   free(rbuffer);
   rbuffer=NULL;

   if (my_rank == rootrank) {
      free(sbuffer);
      sbuffer=NULL;
   }

   MPI_Comm_free(&int_comm);
   MPI_Comm_free(&sub_comm);

   MPI_Finalize();

   return valid_data ? 0 : 1;
}
