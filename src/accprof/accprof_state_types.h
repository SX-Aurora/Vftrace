#ifndef ACCPROF_STATE_TYPES_H
#define ACCPROF_STATE_TYPES_H

#include <stdbool.h>

typedef struct {
   int n_devices;
   char **device_names;
   bool veto_callback_registration;
} accprof_state_t;

#endif
