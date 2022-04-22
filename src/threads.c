#include <stdlib.h>
#include <stdbool.h>

#ifdef _OMP
#include <omp.h>
#endif

#include "realloc_consts.h"
#include "stacks.h"
#include "threads.h"
#include "threadstacks.h"

void vftr_threadtree_realloc(threadtree_t *threadtree_ptr) {
   threadtree_t threadtree = *threadtree_ptr;
   while (threadtree.nthreads > threadtree.maxthreads) {
      int maxthreads = threadtree.maxthreads*vftr_realloc_rate+vftr_realloc_add;
      threadtree.threads = (thread_t*)
         realloc(threadtree.threads, maxthreads*sizeof(thread_t));
      threadtree.maxthreads = maxthreads;
   }
   *threadtree_ptr = threadtree;
}

void vftr_thread_subthreads_realloc(thread_t *thread_ptr) {
   thread_t thread = *thread_ptr;
   while (thread.nsubthreads > thread.maxsubthreads) {
      int maxsubthreads = thread.maxsubthreads*vftr_realloc_rate+vftr_realloc_add;
      thread.subthreads = (int*)
         realloc(thread.subthreads, maxsubthreads*sizeof(int));
      thread.maxsubthreads = maxsubthreads;
   }
   *thread_ptr = thread;
}

int vftr_new_thread(int parent_thread_id,
                    threadtree_t *threadtree_ptr) {
   thread_t thread;
   thread.threadID = threadtree_ptr->nthreads;
   threadtree_ptr->nthreads++;
   vftr_threadtree_realloc(threadtree_ptr);
   thread_t *parent_thread_ptr = threadtree_ptr->threads+parent_thread_id;
   thread.level = parent_thread_ptr->level+1;
   thread.thread_num = parent_thread_ptr->nsubthreads;
   thread.master = parent_thread_ptr->master && thread.thread_num == 0;
   thread.stacklist = vftr_new_threadstacklist(-1);
   thread.parent_thread = parent_thread_ptr->threadID;
   thread.maxsubthreads = 0;
   thread.nsubthreads = 0;
   thread.subthreads = NULL;
   // add thread to threadtree
   threadtree_ptr->threads[thread.threadID] = thread;
   // add it to the subthread list of the parent thread
   parent_thread_ptr->nsubthreads++;

   vftr_thread_subthreads_realloc(parent_thread_ptr);
//printf("parent subthreads = %d\n", parent_thread_ptr->nsubthreads);
   parent_thread_ptr->subthreads[thread.thread_num] = thread.threadID;

   return thread.threadID;
}

thread_t vftr_new_masterthread() {
   thread_t thread;
   thread.level = 0;
   thread.thread_num = 0;
   thread.master = true;
   thread.parent_thread = -1;
   thread.threadID = 0;
   thread.stacklist = vftr_new_threadstacklist(0);
   thread.maxsubthreads = 0;
   thread.nsubthreads = 0;
   thread.subthreads = NULL;
   return thread;
}

threadtree_t vftr_new_threadtree() {
   threadtree_t threadtree;
   threadtree.nthreads = 1;
   threadtree.maxthreads = 1;
   threadtree.threads = (thread_t*) malloc(sizeof(thread_t));
   threadtree.threads[0] = vftr_new_masterthread();
   return threadtree;
}

int vftr_get_thread_level() {
#ifdef _OMP
   return omp_get_level();
#else
   return 0;
#endif
}

int vftr_get_thread_num() {
#ifdef _OMP
   return omp_get_thread_num();
#else
   return 0;
#endif
}

int vftr_get_ancestor_thread_num(int level) {
#ifdef _OMP
   return omp_get_ancestor_thread_num(level);
#else
   (void) level;
   return 0;
#endif
}

thread_t *vftr_get_my_thread(threadtree_t *threadtree_ptr) {
   int level = vftr_get_thread_level();
   thread_t *my_thread = threadtree_ptr->threads;
   // navigate through the thread tree until my thread is found
   for (int ilevel=1; ilevel<level; ilevel++) {
      int thread_num = vftr_get_ancestor_thread_num(ilevel);
      int threadID = my_thread->nsubthreads;
      // if the thread does not exist yet, add and null it.
      while (thread_num > my_thread->nsubthreads) {
         threadID = vftr_new_thread(my_thread->threadID,
                                    threadtree_ptr);
      }
      my_thread = threadtree_ptr->threads+threadID;
   }
   return my_thread;
}

void vftr_thread_free(thread_t *threads_ptr, int threadID) {
   thread_t thread = threads_ptr[threadID];
   vftr_threadstacklist_free(&(thread.stacklist));
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

void vftr_print_thread(FILE *fp, threadtree_t threadtree, int threadid) {
   thread_t thread = threadtree.threads[threadid];
   // first print the indentation
   for (int ilevel=0; ilevel<thread.level; ilevel++) {
      fprintf(fp, "  ");
   }
   fprintf(fp, "%d%s: ", thread.thread_num,
           thread.master ? "m" : "");
   vftr_print_threadstack(fp, thread.stacklist);
   fprintf(fp, "\n");
   for (int ithread=0; ithread<thread.nsubthreads; ithread++) {
      vftr_print_thread(fp, threadtree, thread.subthreads[ithread]);
   }
}

void vftr_print_threadtree(FILE *fp, threadtree_t threadtree) {
   fprintf(fp, "Threadtree:\n");
   vftr_print_thread(fp, threadtree, 0);
}
