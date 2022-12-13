#ifndef ACCPROF_STATE_TYPES_H
#define ACCPROF_STATE_TYPES_H

#include <stdbool.h>

typedef struct open_wait_st {
   long long start_time;
   int async;
   vftr_stack_t *stack;
   struct open_wait_st *next;
} open_wait_t;

typedef struct {
   int n_devices;
   const char **device_names;
   bool veto_callback_registration;
   open_wait_t *open_wait_queues;
   int n_open_wait_queues;
} accprof_state_t;

#endif
