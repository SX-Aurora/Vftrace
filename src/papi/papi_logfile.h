#ifndef PAPI_LOGFILE_H
#define PAPI_LOGFILE_H

#include <stdio.h>

#include "configuration_types.h"
#include "collated_stack_types.h"

void vftr_write_papi_table (FILE *fp, collated_stacktree_t stacktree, config_t config);

void vftr_write_logfile_papi_counter_table (FILE *fp, collated_stacktree_t stacktree, config_t config);

void vftr_write_papi_counter_logfile_summary (FILE *fp, collated_stacktree_t stacktree, config_t config);

void vftr_write_event_descriptions (FILE *fp);
#endif
