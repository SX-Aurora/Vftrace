#ifndef PAPI_LOGFILE_H
#define PAPI_LOGFILE_H

#include <stdio.h>

#include "configuration_types.h"
#include "collated_stack_types.h"

void vftr_write_papi_table (FILE *fp, collated_stacktree_t stacktree, config_t config);

void vftr_write_event_descriptions (FILE *fp);
#endif
