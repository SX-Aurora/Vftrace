#ifndef ACCPROF_LOGFILE_H
#define ACCPROF_LOGFILE_H

#include "vftrace_state.h"
#include "configuration_types.h"
#include "collated_stack_types.h"

char *vftr_name_with_lines_1 (char *name, int line_1);
char *vftr_name_with_lines_2 (char *name, int line_1, int line_2);

void vftr_get_total_accprof_times_for_logfile (collated_stacktree_t stacktree,
                                               double *tot_compute_s, double *tot_memcpy_s, double *tot_other_s);

bool vftr_has_accprof_events (collated_stacktree_t stacktree);

void vftr_write_logfile_accprof_event_table (FILE *fp, collated_stacktree_t stacktree, config_t config);

void vftr_write_accprof_memcpy_stats_all (FILE *fp, collated_stacktree_t stacktree);

void vftr_write_logfile_accprof_grouped_table (FILE *fp, collated_stacktree_t stacktree, config_t config);

void vftr_write_logfile_accev_names (FILE *fp);

#endif
