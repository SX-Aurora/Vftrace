#include <stdlib.h>
#include <stdbool.h>

#include "stacks.h"
#include "threads.h"

thread_t vftr_new_masterthread(stack_t *rootstack_ptr) {
   thread_t thread;
   thread.level = 0;
   thread.thread_num = 0;
   thread.current_stackID = rootstack_ptr->lid;
   thread.master = true;
   thread.parent_thread = -1;
   thread.maxsubthreads = 0;
   thread.nsubthreads = 0;
   thread.subthreads = NULL;
   return thread;
}

threadtree_t vftr_new_threadtree(stack_t *rootstack_ptr) {
   threadtree_t threadtree;
   threadtree.nthreads = 1;
   threadtree.maxthreads = 1;
   threadtree.threads = (thread_t*) malloc(sizeof(thread_t));
   threadtree.threads[0] = vftr_new_masterthread(rootstack_ptr);
   return threadtree;
}

int vftr_get_thread_level() {
   // TODO: use omp_get_level if OMP is active
   return 0;
}

int vftr_get_thread_num() {
   // TODO: use omp_get_thread_num if OMP is active
   return 0;
}

int vftr_get_ancestor_thread_num(int level) {
   (void) level;
   // TODO: use omp_get_ancestor_thread_num(level) if OMP is active
   return 0;
}

thread_t *vftr_get_my_thread(threadtree_t threadtree) {
   int level = vftr_get_thread_level();
   thread_t *my_thread = threadtree.threads;
   // navigate through the thread tree until my thread is found
   for (int ilevel=1; ilevel<level; ilevel++) {
      int thread_num = vftr_get_ancestor_thread_num(ilevel);
      int threadID = my_thread->subthreads[thread_num];
      my_thread = threadtree.threads+threadID;
   }
   return my_thread;
}

void vftr_thread_free(thread_t *threads_ptr, int threadID) {
   thread_t thread = threads_ptr[threadID];
   if (thread.nsubthreads > 0) {
      for (int ithread=0; ithread<thread.nsubthreads; ithread++) {
         vftr_thread_free(threads_ptr, thread.subthreads[ithread]);
      }
      free(thread.subthreads);
      thread.subthreads = NULL;
   }
   threads_ptr[threadID] = thread;
}

void vftr_threadtree_free(threadtree_t *threadtree_ptr) {
   threadtree_t threadtree = *threadtree_ptr;
   if (threadtree.nthreads > 0) {
      vftr_thread_free(threadtree.threads, 0);
      free(threadtree.threads);
      threadtree.threads = NULL;
      threadtree.nthreads = 0;
      threadtree.maxthreads = 0;
   }
   *threadtree_ptr = threadtree;
}
