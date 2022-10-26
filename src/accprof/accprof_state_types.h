#ifndef ACCPROF_STATE_TYPES_H
#define ACCPROF_STATE_TYPES_H

#include <stdbool.h>

typedef struct {
   int n_devices;
   bool veto_callback_registration;
} accprof_state_t;

#endif
