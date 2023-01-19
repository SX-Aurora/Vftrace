#ifndef HWPROF_LOGFILE_H
#define HWPROF_LOGFILE_H

#include <stdio.h>

#include "configuration_types.h"
#include "collated_stack_types.h"

char *vftr_hwtype_string (int hwtype);

void vftr_write_hwprof_observables_table (FILE *fp, collated_stacktree_t stacktree, config_t config);

void vftr_write_logfile_hwprof_counter_table (FILE *fp, collated_stacktree_t stacktree, config_t config);

void vftr_write_hwprof_observables_logfile_summary (FILE *fp, collated_stacktree_t stacktree);

void vftr_write_hwprof_counter_logfile_summary (FILE *fp, collated_stacktree_t stacktree);

#endif
