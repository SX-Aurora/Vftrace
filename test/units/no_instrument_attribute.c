__attribute__((no_instrument_function))
int not_instrumented_function () {
   return 0;
}
int main(int argc, int **argv) {
#ifdef _MPI
   MPI_Init(&argc, &argv);
#endif
   not_instrumented_function();
#ifdef _MPI
   MPI_Finalize();
#endif
}
