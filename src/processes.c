#include "processes.h"
#include "stacks.h"
#include "threads.h"
#include "timer.h"

process_t vftr_new_process() {
   process_t process;
   process.nprocesses = 1;
   process.processID = 0;
   process.stacktree = vftr_new_stacktree();
   process.threadtree = vftr_new_threadtree(process.stacktree.stacks);

   return process;
}

void vftr_process_free(process_t *process_ptr) {
   process_t process = *process_ptr;
   vftr_stacktree_free(&(process.stacktree));
   vftr_threadtree_free(&(process.threadtree));
}
