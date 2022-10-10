#ifndef LOGFILE_CUPTI_TABLE_H
#define LOGFILE_CUPTI_TABLE_H

void vftr_get_total_cupti_times (collated_stacktree_t stacktree,
                                 float *tot_compute_s, float *tos_memcpy_s, float *tot_other_s);
void vftr_write_logfile_cupti_table(FILE *fp, collated_stacktree_t stacktree);

#endif
