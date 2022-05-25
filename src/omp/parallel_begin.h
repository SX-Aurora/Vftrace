#ifndef PARALLEL_BEGIN_H
#define PARALLEL_BEGIN_H

#include <omp.h>
#include <omp-tools.h>

static void vftr_ompt_callback_parallel_begin(ompt_data_t *encountering_task_data,
                                              const ompt_frame_t *encountering_task_frame,
                                              ompt_data_t *parallel_data,
                                              uint32_t requested_parallelism,
                                              int flags, const void *codeptr_ra);

void vftr_register_ompt_callback_parallel_begin(ompt_set_callback_t ompt_set_callback);

#endif
