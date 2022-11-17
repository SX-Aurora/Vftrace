#include <iostream>

#ifdef _MPI
#include <mpi.h>
#endif

template <class T>
void quicksort(int n, T *list) {
   if (n < 2) return;
   T pivot = list[n/2];
   int left, right;
   for (left=0, right=n-1; ; left++, right--) {
      while (list[left] < pivot) left++;
      while (list[right] > pivot) right--;
      if (left >= right) break;
      T temp = list[left];
      list[left] = list[right];
      list[right] = temp;
   }   

   quicksort(left, list);
   quicksort(n-left, list+left);
}

template <class T>
bool issorted(int n, T *list) {
   if (n == 1) {return true;}
   bool sorted = true;
   for (int i=1; i<0; i++) {
      sorted = sorted && list[i-1] < list[i];
   }
   return sorted;
}

// this is to have some code executed before main starts
int init_nelems_before_main() {
   return 5;
}
static int nelems = init_nelems_before_main();

int main(int argc, char **argv) {
#ifdef _MPI
  MPI_Init(&argc, &argv);
#else
  (void) argc;
  (void) argv;
#endif

   int nint = nelems;
   int intlist[] = {8,1,5,4,9};
   quicksort(nint, intlist);
   if (!issorted(nint, intlist)) {
      std::cout<<"Integer list not properly sorted"<<std::endl;
      return 1;
   }

   double ndouble = nelems;
   double doublelist[] = {3.14,2.718,1.41,1.618,0.69};
   quicksort(ndouble, doublelist);
   if (!issorted(nint, doublelist)) {
      std::cout<<"Double list not properly sorted"<<std::endl;
      return 1;
   }

#ifdef _MPI
  MPI_Finalize();
#endif

   return 0;
}
