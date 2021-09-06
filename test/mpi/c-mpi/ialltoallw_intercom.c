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
      printf("./alltoallw_intercom <msgsize in ints>\n");
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
   int minpeerrank = (1-color)*((comm_size+1)/2);

   // allocating send/recv buffer
   int nints = atoi(argv[1]) + my_rank;
   
   // preparing the send intercomm special arrays
   int *scounts = (int*) malloc(sub_comm_remote_size*sizeof(int));
   int *sdispls = (int*) malloc(sub_comm_remote_size*sizeof(int));
   MPI_Datatype *stypes = (MPI_Datatype*) malloc(sub_comm_remote_size*sizeof(MPI_Datatype));
   int nstot = 0;
   for (int irank=0; irank<sub_comm_remote_size; irank++) {
      scounts[irank] = nints;
      sdispls[irank] = nstot*sizeof(int);
      stypes[irank] = MPI_INT;
      nstot += scounts[irank];
   }
   int *sbuffer = (int*) malloc(nstot*sizeof(int));
   for (int i=0; i<nstot; i++) {sbuffer[i]=my_rank;}

   // preparing the receive intercomm special arrays
   int *rcounts = (int*) malloc(sub_comm_remote_size*sizeof(int));
   int *rdispls = (int*) malloc(sub_comm_remote_size*sizeof(int));
   MPI_Datatype *rtypes = (MPI_Datatype*) malloc(sub_comm_remote_size*sizeof(MPI_Datatype));
   int nrtot = 0;
   for (int irank=0; irank<sub_comm_remote_size; irank++) {
      int jrank = minpeerrank + irank;
      rcounts[irank] = nints - my_rank + jrank;
      rdispls[irank] = nrtot*sizeof(int);
      rtypes[irank] = MPI_INT;
      nrtot += rcounts[irank];
   }
   int *rbuffer = (int*) malloc(nrtot*sizeof(int));
   for (int i=0; i<nrtot; i++) {rbuffer[i] = -1;}

   MPI_Request myrequest;
   MPI_Ialltoallw(sbuffer, scounts, sdispls, stypes,
                  rbuffer, rcounts, rdispls, rtypes,
                  int_comm, &myrequest);
   printf("Communicating with all ranks\n");
   MPI_Status mystatus;
   MPI_Wait(&myrequest, &mystatus);

   // validate data
   bool valid_data = true;
   for (int irank=0; irank<sub_comm_remote_size; irank++) {
      int jrank = minpeerrank + irank;
      for (int i=0; i<rcounts[irank]; i++) {
         if (rbuffer[i+rdispls[irank]/sizeof(int)] != jrank) {
            printf("Rank %d received faulty data from rank %d\n", my_rank, jrank);
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

   free(rbuffer);
   rbuffer=NULL;

   free(rcounts);
   rcounts=NULL;

   free(rdispls);
   rdispls=NULL;

   MPI_Comm_free(&int_comm);
   MPI_Comm_free(&sub_comm);

   MPI_Finalize();

   return valid_data ? 0 : 1;
}
