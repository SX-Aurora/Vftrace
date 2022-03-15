#include "stack_types.h"

long long vftr_total_overhead_usec(stacktree_t stacktree) {
   long long overhead = 0;
   for (int istack=0; istack<stacktree.nstacks; istack++) {
      overhead += stacktree.stacks[istack].profiling.callProf.overhead_time_usec;
   }
   return overhead;
}
