#include <stdlib.h>
#include <stdbool.h>

#ifdef _OMP
#include <omp.h>
#endif

#include "self_profile.h"
#include "realloc_consts.h"
#include "stacks.h"
#include "threads.h"
#include "threadstacks.h"

void vftr_threadtree_realloc(threadtree_t *threadtree_ptr) {
   SELF_PROFILE_START_FUNCTION;
   threadtree_t threadtree = *threadtree_ptr;
   while (threadtree.nthreads > threadtree.maxthreads) {
      int maxthreads = threadtree.maxthreads*vftr_realloc_rate+vftr_realloc_add;
      threadtree.threads = (thread_t*)
         realloc(threadtree.threads, maxthreads*sizeof(thread_t));
      threadtree.maxthreads = maxthreads;
   }
   *threadtree_ptr = threadtree;
   SELF_PROFILE_END_FUNCTION;
}

void vftr_thread_subthreads_realloc(thread_t *thread_ptr) {
   SELF_PROFILE_START_FUNCTION;
   thread_t thread = *thread_ptr;
   while (thread.nsubthreads > thread.maxsubthreads) {
      int maxsubthreads = thread.maxsubthreads*vftr_realloc_rate+vftr_realloc_add;
      thread.subthreads = (int*)
         realloc(thread.subthreads, maxsubthreads*sizeof(int));
      thread.maxsubthreads = maxsubthreads;
   }
   *thread_ptr = thread;
   SELF_PROFILE_END_FUNCTION;
}

int vftr_new_thread(int parent_thread_id,
                    threadtree_t *threadtree_ptr) {
   SELF_PROFILE_START_FUNCTION;
   thread_t thread;
   thread.threadID = threadtree_ptr->nthreads;
   threadtree_ptr->nthreads++;
   vftr_threadtree_realloc(threadtree_ptr);
   thread_t *parent_thread_ptr = threadtree_ptr->threads+parent_thread_id;
   thread.level = parent_thread_ptr->level+1;
   thread.thread_num = parent_thread_ptr->nsubthreads;
   thread.master = parent_thread_ptr->master && thread.thread_num == 0;
   // inherit stackID from parent thread
   threadstack_t *parent_threadstack = vftr_get_my_threadstack(parent_thread_ptr);
   thread.stacklist = vftr_new_threadstacklist(parent_threadstack->stackID);
   thread.parent_thread = parent_thread_ptr->threadID;
   thread.maxsubthreads = 0;
   thread.nsubthreads = 0;
   thread.subthreads = NULL;
   // add thread to threadtree
   threadtree_ptr->threads[thread.threadID] = thread;
   // add it to the subthread list of the parent thread
   parent_thread_ptr->nsubthreads++;

   vftr_thread_subthreads_realloc(parent_thread_ptr);
   parent_thread_ptr->subthreads[thread.thread_num] = thread.threadID;

   SELF_PROFILE_END_FUNCTION;
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
   SELF_PROFILE_START_FUNCTION;
   threadtree_t threadtree;
   threadtree.nthreads = 1;
   threadtree.maxthreads = 1;
   threadtree.threads = (thread_t*) malloc(sizeof(thread_t));
   threadtree.threads[0] = vftr_new_masterthread();
   SELF_PROFILE_END_FUNCTION;
   return threadtree;
}

void vftr_thread_subthreads_reset(threadtree_t *threadtree_ptr, int threadID) {
   thread_t *my_thread = threadtree_ptr->threads+threadID;
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);
   int nsubthreads = my_thread->nsubthreads;
   for (int isubthread=0; isubthread<nsubthreads; isubthread++) {
      int subthreadID = my_thread->subthreads[isubthread];
      thread_t *thread = threadtree_ptr->threads+subthreadID;
      thread->stacklist.nstacks = 0;
      vftr_threadstack_push(my_threadstack->stackID, &(thread->stacklist));
   }
}

int vftr_get_thread_level() {
   SELF_PROFILE_START_FUNCTION;
#ifdef _OMP
   int level = omp_get_level();
#else
   int level = 0;
#endif
   SELF_PROFILE_END_FUNCTION;
   return level;
}

int vftr_get_thread_num() {
   SELF_PROFILE_START_FUNCTION;
#ifdef _OMP
   int thread_num = omp_get_thread_num();
#else
   int thread_num = 0;
#endif
   SELF_PROFILE_END_FUNCTION;
   return thread_num;
}

int vftr_get_ancestor_thread_num(int level) {
   SELF_PROFILE_START_FUNCTION;
#ifdef _OMP
   int ancestor_thread_num = omp_get_ancestor_thread_num(level);
#else
   (void) level;
   int ancestor_thread_num = 0;
#endif
   SELF_PROFILE_END_FUNCTION;
   return ancestor_thread_num;
}

thread_t *vftr_get_my_thread(threadtree_t *threadtree_ptr) {
   SELF_PROFILE_START_FUNCTION;
   int level = vftr_get_thread_level();
   int parent_threadID = -1;
   int threadID = 0;
   thread_t *parent_thread = NULL;
   // navigate through the thread tree until my thread is found
   for (int ilevel = 1; ilevel <= level; ilevel++) {
      parent_threadID = threadID;
      int thread_num = vftr_get_ancestor_thread_num(ilevel);
      while (thread_num >= (threadtree_ptr->threads+parent_threadID)->nsubthreads) {
         threadID = vftr_new_thread(parent_threadID,
                                    threadtree_ptr);
      }
      parent_thread = threadtree_ptr->threads+parent_threadID;
      threadID = parent_thread->subthreads[thread_num];
   }
   thread_t *my_thread = threadtree_ptr->threads + threadID;
   SELF_PROFILE_END_FUNCTION;
   return my_thread;
}

void vftr_thread_free(thread_t *threads_ptr, int threadID) {
   SELF_PROFILE_START_FUNCTION;
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
   SELF_PROFILE_END_FUNCTION;
}

void vftr_threadtree_free(threadtree_t *threadtree_ptr) {
   SELF_PROFILE_START_FUNCTION;
   threadtree_t threadtree = *threadtree_ptr;
   if (threadtree.nthreads > 0) {
      vftr_thread_free(threadtree.threads, 0);
      free(threadtree.threads);
      threadtree.threads = NULL;
      threadtree.nthreads = 0;
      threadtree.maxthreads = 0;
   }
   *threadtree_ptr = threadtree;
   SELF_PROFILE_END_FUNCTION;
}

void vftr_print_thread(FILE *fp, threadtree_t threadtree, int threadid) {
   thread_t thread = threadtree.threads[threadid];
   // first print the indentation
   for (int ilevel=0; ilevel<thread.level; ilevel++) {
      fprintf(fp, "  ");
   }
   fprintf(fp, "%4d%s", thread.thread_num,
           thread.master ? "m" : "");
   fprintf(fp, "\n");
   for (int ithread=0; ithread<thread.nsubthreads; ithread++) {
      vftr_print_thread(fp, threadtree, thread.subthreads[ithread]);
   }
}

void vftr_print_current_thread(FILE *fp, threadtree_t threadtree, int threadid) {
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

void vftr_print_threadlist(FILE *fp, threadtree_t threadtree) {
   for (int ithread=0; ithread<threadtree.nthreads; ithread++) {
      thread_t *thread = threadtree.threads+ithread;
      fprintf(fp, "%4d:", thread->threadID);
      if (thread->parent_thread >= 0) {
         fprintf(fp, " parent=%d", thread->parent_thread);
      } else {
         fprintf(fp, " parent=-");
      }
      fprintf(fp, ", lvl=%d", thread->level);
      if (thread->nsubthreads > 0)  {
         fprintf(fp, ", subthreads=%d", thread->subthreads[0]);
         for (int isubthread=1; isubthread<thread->nsubthreads; isubthread++) {
            fprintf(fp, ",%d", thread->subthreads[isubthread]);
         }
      }
      fprintf(fp, "\n");
   }
}
