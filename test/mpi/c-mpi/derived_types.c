#include <mpi.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <unistd.h>

typedef struct teststruct_t {
   int anint1, anint2;
   char string[16];
   double adouble1, adouble2, adouble3;
   int intarr[32];
} teststruct;

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

   // allocate srbuffer
   const int nelem = 1024;
   teststruct *srbuffer = (teststruct*) malloc(nelem*sizeof(teststruct));

   // construct the MPI-Datatypes for a nested construct
   // datatype for the char arry of size 16
   MPI_Datatype mpi_chararr_t;
   MPI_Type_contiguous(16, MPI_CHAR, &mpi_chararr_t);
   MPI_Type_commit(&mpi_chararr_t);
   // datatype for the int array of size 32
   MPI_Datatype mpi_intarr_t;
   MPI_Type_contiguous(32, MPI_INT, &mpi_intarr_t);
   MPI_Type_commit(&mpi_intarr_t);
   // Create the struct type
   int blocklength[4] = {2,1,3,1};
   MPI_Aint displacements[4] = {0,8,24,48};
   MPI_Datatype types[4] = {MPI_INT, mpi_chararr_t, MPI_DOUBLE, mpi_intarr_t};
   MPI_Datatype mpi_teststruct_t;
   MPI_Type_create_struct(4, blocklength, displacements, types, &mpi_teststruct_t);
   MPI_Type_commit(&mpi_teststruct_t);

   // Messaging 
   if (my_rank == 0) {
      // sending rank
      MPI_Send(srbuffer, nelem, mpi_teststruct_t, 1, 0, MPI_COMM_WORLD);
   } else if (my_rank == 1) {
      // receiving rank
      MPI_Status recvstatus;
      MPI_Recv(srbuffer, nelem, mpi_teststruct_t, 0, 0, MPI_COMM_WORLD, &recvstatus);
   }

   MPI_Type_free(&mpi_teststruct_t);
   MPI_Type_free(&mpi_intarr_t);
   MPI_Type_free(&mpi_chararr_t);

   // deallocate the buffer
   free(srbuffer);
   srbuffer = NULL;

   MPI_Finalize();

   return 0;
}
