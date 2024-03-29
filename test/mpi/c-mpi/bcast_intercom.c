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
      printf("./bcast_intercom <msgsize in ints>\n");
      return 1;
   }

   // allocating send/recv buffer
   int nints = atoi(argv[1]);
   int *buffer = (int*) malloc(nints*sizeof(int));
   for (int i=0; i<nints; i++) {buffer[i]=my_rank;}

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

   // preparing the intercomm root assignment
   int root;
   if (color == 0) {
      // Sub communicator of sending group
      if (my_sub_rank == 0) {
         // Sending rank in sending subgroup
         root = MPI_ROOT;
      } else {
         // Ideling processes
         root = MPI_PROC_NULL;
      }
   } else {
      // Sub communicator or receiving group
      root = 0; // sending rank in remote group
   }
   MPI_Bcast(buffer, nints, MPI_INT, root, int_comm);
   if (my_rank == 0) {
      printf("Broadcasted message from global rank %d\n", my_rank);
      printf("(Group=%d, local rank=%d)\n", color, my_sub_rank);
   }

   // validate data
   bool valid_data = true;
   switch(root) {
      case MPI_ROOT:
         for (int i=0; i<nints; i++) {
            if (buffer[i] != my_rank) {
               printf("Root process (Rank %d) has no longer valid data\n", my_rank);
               valid_data = false;
               break;
            }
         }
         break;
      case MPI_PROC_NULL:
         for (int i=0; i<nints; i++) {
            if (buffer[i] != my_rank) {
               printf("Uninvolved process (Rank %d) has no longer valid data\n", my_rank);
               valid_data = false;
               break;
            }
         }
         break;
      default:
         for (int i=0; i<nints; i++) {
            if (buffer[i] != root) {
               printf("Rank %d received faulty data from rank %d\n", my_rank, root);
               valid_data = false;
               break;
            }
         }
         break;
   }

   MPI_Comm_free(&int_comm);
   MPI_Comm_free(&sub_comm);

   free(buffer);
   buffer=NULL;

   MPI_Finalize();

   return valid_data ? 0 : 1;
}
