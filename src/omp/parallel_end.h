#ifndef PARALLEL_END_H
#define PARALLEL_END_H

#include <omp.h>
#include <omp-tools.h>

static void vftr_ompt_callback_parallel_end(ompt_data_t *parallel_data,
                                            ompt_data_t *encountering_task_data,
                                            int flags, const void *codeptr_ra);

void vftr_register_ompt_callback_parallel_end(ompt_set_callback_t ompt_set_callback);

#endif
