#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

#ifdef _REGIONS
#include <vftrace.h>
#endif

#ifdef _MPI
#include <mpi.h>
#endif

int collatz_max_steps(long long nmax) {
   int max_steps = 0;
   for (long long n=1; n<=nmax; n++) {
      long long i = n;
      int steps = 0;
      while (i != 1) {
         i = (i&1) ? (3*i+1) : (i/2);
         steps++;
      }
      if (steps > max_steps) {
         max_steps = steps;
      }
   }
   return max_steps;
}

int pythagorean_triples(int n) {
   int largest = 0;
   for (int a=1; a<n; a++) {
      for (int b=a; b<n; b++) {
         int csq = a*a + b*b;
         int c = (int) (sqrt(csq)+0.5);
         if (c*c == csq && csq > largest) {
            largest = csq;
         }
      }
   }
   return largest;
}


int largest_prime(int n) {
   if (n<2) {return -1;}
   if (n==2) {return 2;}
   int largest = 0;
   for (int i=3; i<=n; i+=2) {
      bool is_prime = true;
      int tmax = (int) sqrt(1.0*i);
      for (int t=3; t<=tmax; t+=2) {
         if (i%t == 0) {
            is_prime = false;
            break;
         }
      }
      if (is_prime) {
         largest = i;
      }
   }
   return largest;
}

int main(int argc, char **argv) {
#ifdef _MPI
  MPI_Init(&argc, &argv);
#else
  (void) argc;
  (void) argv;
#endif

  // collatz conjecture
  // longest chain below 1e6
  int collatz = collatz_max_steps(100);
  // pythagoras
  int pythagoras = pythagorean_triples(100);
  // largest prime below 5e6
  int prime = largest_prime(100);

#ifdef _MPI
  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  if (myrank == 0) {
#endif
  printf("collatz: %d\n", collatz);
  printf("pythagoras: %d\n", pythagoras);
  printf("prime: %d\n", prime);
#ifdef _MPI
  }
#endif

#ifdef _REGIONS
  vftrace_region_begin("MyRegion");
  vftrace_region_end("MyRegion");
#endif


#ifdef _MPI
  MPI_Finalize();
#endif

  return 0;
}
