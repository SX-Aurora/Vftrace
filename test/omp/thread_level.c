#include <stdio.h>
#include <stdbool.h>

#include <omp.h>

#include "threads.h"

int main(int argc, char **argv) {
   int level = vftr_get_thread_level();
   int thread = vftr_get_thread_num();
   printf("Thread %d at level %d\n", thread, level);
   #pragma omp parallel num_threads(2) private(level,thread)
   {
      level = vftr_get_thread_level();
      thread = vftr_get_thread_num();
      printf("Thread %d at level %d\n", thread, level);
      if (thread == 1) {
         #pragma omp parallel num_threads(2) private(level,thread)
         {
            level = vftr_get_thread_level();
            thread = vftr_get_thread_num();
            printf("Thread %d at level %d\n", thread, level);
         }
      }
   }

   return 0;
}
