#include <mpi.h>

#include <stdlib.h>

int main (int argc, char *argv[]) {
    MPI_Init (&argc, &argv);
    int *a = (int*)malloc(sizeof(int));

    #pragma acc enter data copyin(a)
    #pragma acc exit data copyout(a)

    free (a);
    MPI_Finalize();
}
