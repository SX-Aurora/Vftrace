#ifndef PAPI_RANKLOGFILE_H
#define PAPI_RANKLOGFILE_H

#include <stdio.h>

#include "configuration_types.h"
#include "stack_types.h"

void vftr_write_ranklogfile_papi_obs_table (FILE *fp, stacktree_t stacktree, config_t config);

void vftr_write_ranklogfile_papi_counter_table (FILE *fp, stacktree_t stacktree, config_t config);

void vftr_write_papi_observables_ranklogfile_summary (FILE *fp, stacktree_t stacktree, config_t config);

void vftr_write_papi_counter_ranklogfile_summary (FILE *fp, stacktree_t stacktree, config_t config);

#endif
