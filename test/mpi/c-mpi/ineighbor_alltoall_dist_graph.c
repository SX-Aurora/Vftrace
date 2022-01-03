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
      printf("./ineighbor_alltoall_dist_graph <msgsize in ints>\n");
      return 1;
   }

   // requires precicely 4 processes
   if (comm_size != 4) {
      printf("requires precicely 4 processes. Start with -np 4!\n");
      return 1;
   }

   // Create the distributed graph communicator
   MPI_Comm comm_dist_graph;
   const int nnodes = 1;
   int sources[1] = {my_rank};
   int degrees[1];
   int *destinations = NULL;
   int reorder = false;
   switch(my_rank) {
      case 0:
         degrees[0] = 3;
         destinations = (int*) malloc(degrees[0]*sizeof(int));
         destinations[0] = 1;
         destinations[1] = 1;
         destinations[2] = 2;
         break;
      case 1:
         degrees[0] = 3;
         destinations = (int*) malloc(degrees[0]*sizeof(int));
         destinations[0] = 0;
         destinations[1] = 0;
         destinations[2] = 2;
         break;
      case 2:
         degrees[0] = 2;
         destinations = (int*) malloc(degrees[0]*sizeof(int));
         destinations[0] = 0;
         destinations[1] = 3;
         break;
      case 3:
         degrees[0] = 1;
         destinations = (int*) malloc(degrees[0]*sizeof(int));
         destinations[0] = 3;
         break;
   }
   MPI_Dist_graph_create(MPI_COMM_WORLD, nnodes, sources,
                         degrees, destinations,
                         MPI_UNWEIGHTED, MPI_INFO_NULL,
                         reorder, &comm_dist_graph);
   free(destinations);

   int indegree;
   int outdegree;
   int weighted;
   MPI_Dist_graph_neighbors_count(comm_dist_graph, &indegree, &outdegree, &weighted);
   int *inneighbors = (int*) malloc(indegree*sizeof(int));
   int *inweights = (int*) malloc(indegree*sizeof(int));
   int *outneighbors = (int*) malloc(outdegree*sizeof(int));
   int *outweights = (int*) malloc(outdegree*sizeof(int));
   MPI_Dist_graph_neighbors(comm_dist_graph,
                            indegree, inneighbors, inweights,
                            outdegree, outneighbors, outweights);

   // allocating send/recv buffer
   int nints = atoi(argv[1]);
   int *sbuffer = (int*) malloc(nints*outdegree*sizeof(int));
   for (int i=0; i<nints*outdegree; i++) {sbuffer[i]=my_rank;}
   int *rbuffer = (int*) malloc(nints*indegree*sizeof(int));
   for (int i=0; i<nints*indegree; i++) {rbuffer[i]=-1;}

   MPI_Request request;
   MPI_Ineighbor_alltoall(sbuffer, nints, MPI_INT,
                          rbuffer, nints, MPI_INT,
                          comm_dist_graph, &request);
   MPI_Status status;
   MPI_Wait(&request, &status);

   // validate data
   bool valid_data = true;
   for (int ineighbor=0; ineighbor<indegree; ineighbor++) {
      int refval = inneighbors[ineighbor];
      for (int i=0; i<nints; i++) {
         if (rbuffer[i+ineighbor*nints] != refval) {
            printf("Rank %d received faulty data from rank %d\n", my_rank, refval);
            valid_data = false;
            break;
         }
      }
   }

   free(inneighbors);
   inneighbors = NULL;
   free(inweights);
   inweights = NULL;
   free(outneighbors);
   outneighbors = NULL;
   free(outweights);
   outweights = NULL;
   free(rbuffer);
   rbuffer=NULL;
   free(sbuffer);
   sbuffer=NULL;

   MPI_Comm_free(&comm_dist_graph);
   MPI_Finalize();

   return valid_data ? 0 : 1;
}
