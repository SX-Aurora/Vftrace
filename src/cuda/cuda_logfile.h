#ifndef CUDA_LOGFILE_H
#define CUDA_LOGFILE_H

#include "configuration_types.h"

void vftr_get_total_cuda_times_for_logfile (collated_stacktree_t stacktree,
                                             float *tot_compute_s, float *tos_memcpy_s, float *tot_other_s);
void vftr_write_logfile_cuda_table (FILE *fp, collated_stacktree_t stacktree, config_t config);

void vftr_write_cuda_memcpy_stats_all (FILE *fp, collated_stacktree_t stacktree);

void vftr_write_logfile_cbid_names (FILE *fp, collated_stacktree_t stacktree);

#endif
