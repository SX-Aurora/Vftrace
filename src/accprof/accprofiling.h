#ifndef ACCPROFILING_H
#define ACCPROFILING_H

#include "accprofiling_types.h"

accprofile_t vftr_new_accprofiling();

void vftr_accumulate_accprofiling (accprofile_t *prof, acc_event_t ev,
                                   int line_start, int line_end,
  				   const char *source_file, const char *func_name,
                                   const char *kernel_name, const char *var_name, size_t copied_bytes);

void vftr_accumulate_accprofiling_overhead (accprofile_t *prof, long long t_nsec);

accprofile_t vftr_add_accprofiles (accprofile_t profA, accprofile_t profB);

long long vftr_get_total_accprof_overhead (stacktree_t stacktree);
long long vftr_get_toal_collated_accprof_overhead (collated_stacktree_t stacktree);

void vftr_accprofiling_free (accprofile_t *prof_ptr);

#endif
