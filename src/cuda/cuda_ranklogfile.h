#ifndef RANKLOGFILE_CUDA_TABLE_H
#define RANKLOGFILE_CUDA_TABLE_H

#include "configuration_types.h"

void vftr_get_total_cuda_times_for_ranklogfile (stacktree_t stacktree, 
                                             float *tot_compute_s, float *tot_memcpy_s, float *tot_other_s);

void vftr_write_ranklogfile_cuda_table(FILE *fp, stacktree_t stacktree, config_t config);

void vftr_write_cuda_memcpy_stats (FILE *fp, stacktree_t stacktree, config_t config);

void vftr_write_ranklogfile_cbid_names (FILE *fp, stacktree_t stacktree);

#endif
