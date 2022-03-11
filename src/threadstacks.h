#ifndef THREADSTACKS_H
#define THREADSTACKS_H

#include "threadstack_types.h"

void vftr_threadstacklist_realloc(threadstacklist_t *stacklist_ptr);

threadstacklist_t vftr_new_threadstacklist();

void vftr_threadstack_free(threadstack_t *stack_ptr);

// push a new callstack onto the threads local stack
void vftr_threadstack_push(int stackID, threadstacklist_t *stacklist_ptr);

threadstack_t vftr_threadstack_pop(threadstacklist_t *stacklist_ptr);

void vftr_threadstacklist_free(threadstacklist_t *stacklist_ptr);

threadstack_t *vftr_get_my_threadstack(thread_t *my_thread_ptr);

#ifdef _DEBUG
void vftr_print_threadstack(FILE *fp, threadstacklist_t stacklist);
#endif

#endif
