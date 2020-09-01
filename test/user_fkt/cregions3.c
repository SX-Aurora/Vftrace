#include <stdio.h>
#include <stdlib.h>
#ifdef _MPI
#include <mpi.h>
#endif

#include <vftrace.h>

int ndigits(int number) {
   int n = 1;
   while (number >= 10) {
      number /= 10;
      n++;
   }
   return n;
}

int main(int argc, char** argv) {

#ifdef _MPI
   MPI_Init(NULL, NULL);
#endif

   // require cmd-line argument
   if (argc < 2) {
      printf("./cregions3 <nregions>\n");
      return 1;
   }

   int nreg = atoi(argv[1]);

   for (int ireg=1; ireg<=nreg; ireg++) {
      char reg_name[32];
      sprintf(reg_name, "user-region-%*d", ndigits(ireg), ireg);
      printf("%s\n", reg_name);
      vftrace_region_begin(reg_name);
      vftrace_region_end(reg_name);
   }

#ifdef _MPI
   MPI_Finalize();
#endif
}

