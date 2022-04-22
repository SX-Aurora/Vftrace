#include <stdio.h>
#include <stdbool.h>

#include <omp.h>

#include "threads.h"

int main(int argc, char **argv) {
   int level = vftr_get_thread_level();
   int thread = vftr_get_thread_num();
   int ancestor = 0;
   printf("Thread %d at level %d\n", thread, level);
   #pragma omp parallel num_threads(2) private(level,thread,ancestor)
   {
      level = vftr_get_thread_level();
      thread = vftr_get_thread_num();
      ancestor = vftr_get_ancestor_thread_num(level-1);
      printf("Thread %d at level %d with ancestor %d\n", thread, level, ancestor);
      if (thread == 1) {
         #pragma omp parallel num_threads(2) private(level,thread,ancestor)
         {
            level = vftr_get_thread_level();
            thread = vftr_get_thread_num();
            ancestor = vftr_get_ancestor_thread_num(level-1);
            printf("Thread %d at level %d with ancestor %d\n", thread, level, ancestor);
         }
      }
   }

   return 0;
}
