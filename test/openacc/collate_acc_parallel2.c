#include <mpi.h>

#include <stdlib.h>

int main (int argc, char *argv[]) {
    MPI_Init (&argc, &argv);

    int rank;
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    int alloc_size = rank == 0 ? 1 : 2;
    int a[alloc_size];

    #pragma acc enter data copyin(a)
    #pragma acc exit data copyout(a)

    MPI_Finalize();
}
