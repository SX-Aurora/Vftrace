#ifndef ACCPROF_EVENTS_H
#define ACCPROF_EVENTS_H

#include <stdbool.h>

#include "acc_prof.h"

bool vftr_accprof_is_data_event (acc_event_t event_type);
bool vftr_accprof_is_compute_event (acc_event_t event_type);
char *vftr_accprof_event_string (acc_event_t event_type);

bool vftr_accprof_is_h2d_event (acc_event_t event_type);
bool vftr_accprof_is_d2h_event (acc_event_t event_type);
bool vftr_accprof_is_ondevice_event (acc_event_t event_type);
bool vftr_accprof_is_launch_event (acc_event_t event_type);
bool vftr_accprof_is_defined (acc_event_t event_type);

#endif
