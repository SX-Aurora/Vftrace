#ifndef ACCPROF_LOGFILE_H
#define ACCPROF_LOGFILE_H

#include "configuration_types.h"
#include "collated_stack_types.h"

char *vftr_name_with_lines (char *name, int line_1, int line_2);

void vftr_get_total_accprof_times_for_logfile (collated_stacktree_t stacktree,
                                               double *tot_compute_s, double *tot_memcpy_s, double *tot_other_s);

void vftr_write_logfile_accprof_table (FILE *fp, collated_stacktree_t stacktree, config_t config);

void vftr_write_logfile_accev_names (FILE *fp);

#endif
