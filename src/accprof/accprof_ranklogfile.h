#ifndef ACCPROF_RANKLOGFILE_H
#define ACCPROF_RANKLOGFILE_H

#include "configuration_types.h"
#include "stack_types.h"

void vftr_get_total_accprof_times_for_ranklogfile (stacktree_t stacktree, double *tot_compute_s,
						   double *tot_memcpy_s, double *tot_wait_s);
void vftr_write_ranklogfile_accprof_event_table (FILE *fp, stacktree_t stacktree, config_t config);

void vftr_write_ranklogfile_accprof_grouped_table (FILE *fp, stacktree_t stacktree, config_t config);
#endif
