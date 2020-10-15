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
      printf("./reduce_intercom <msgsize in ints>\n");
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

   // allocating send/recv buffer
   int nints = atoi(argv[1]);
   
   int *sbuffer = (int*) malloc(nints*sizeof(int));
   for (int i=0; i<nints; i++) {sbuffer[i]=my_rank;}
   int *rbuffer = NULL;
   if (my_rank == 0) {
      rbuffer = (int*) malloc(nints*sizeof(int));
      for (int i=0; i<nints; i++) {
         rbuffer[i] = -1;
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

   MPI_Request myrequest;
   MPI_Ireduce(sbuffer, rbuffer, nints, MPI_INT,
               MPI_SUM, root, int_comm, &myrequest);
   if (my_rank == 0) {
      printf("Reducing message on global rank %d\n", my_rank);
      printf("(Group=%d, local rank=%d)\n", color, my_sub_rank);
   }
   MPI_Status mystatus;
   MPI_Wait(&myrequest, &mystatus);

   // validate data
   bool valid_data = true;
   if (my_rank == 0) {
      int refresult = (comm_size-1)*comm_size;
      refresult -= (sub_comm_size-1)*sub_comm_size;
      refresult /= 2;
      for (int i=0; i<nints; i++) {
         if (rbuffer[i] != refresult) {
            printf("Rank %d received faulty data\n", my_rank);
            valid_data = false;
            break;
         }
      }
      free(rbuffer);
      rbuffer=NULL;
   }

   MPI_Comm_free(&int_comm);
   MPI_Comm_free(&sub_comm);

   free(sbuffer);
   sbuffer=NULL;

   MPI_Finalize();

   return valid_data ? 0 : 1;
}
