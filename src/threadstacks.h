#ifndef THREADSTACKS_H
#define THREADSTACKS_H

#include <stdio.h>
#include <stdbool.h>

#include "thread_types.h"
#include "threadstack_types.h"
#include "vftrace_state.h"

void vftr_threadstacklist_realloc(threadstacklist_t *stacklist_ptr);

threadstacklist_t vftr_new_threadstacklist(int stackID);

void vftr_threadstack_free(threadstack_t *stack_ptr);

// push a new callstack onto the threads local stack
void vftr_threadstack_push(int stackID, threadstacklist_t *stacklist_ptr);

threadstack_t *vftr_threadstack_pop(threadstacklist_t *stacklist_ptr);

void vftr_threadstacklist_free(threadstacklist_t *stacklist_ptr);

threadstack_t *vftr_get_my_threadstack(thread_t *my_thread_ptr);

threadstack_t *vftr_update_threadstack_function(threadstack_t *my_threadstack,
                                                thread_t *my_thread,
                                                uintptr_t func_addr,
                                                vftrace_t *vftrace);

threadstack_t *vftr_update_threadstack_region(threadstack_t *my_threadstack,
                                              thread_t *my_thread,
                                              uintptr_t region_addr,
                                              const char *name,
                                              vftrace_t *vftrace,
                                              bool precise);


void vftr_print_threadstack(FILE *fp, threadstacklist_t stacklist);

#endif
