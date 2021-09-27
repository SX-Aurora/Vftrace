#ifdef _MPI
#include <mpi.h>
#endif
__attribute__((no_instrument_function))
int not_instrumented_function () {
   return 0;
}
int instrumented_function () {
   return 0;
}
int main(int argc, char **argv) {
#ifdef _MPI
   MPI_Init(&argc, &argv);
#endif
   not_instrumented_function();
   instrumented_function();
#ifdef _MPI
   MPI_Finalize();
#endif
}
