#ifndef IMPLICIT_TASK_H
#define IMPLICIT_TASK_H

#include <omp.h>
#include <omp-tools.h>

static void vftr_ompt_callback_implicit_task(ompt_scope_endpoint_t endpoint,
                                             ompt_data_t *parallel_data,
                                             ompt_data_t *task_data,
                                             unsigned int actual_parallelism,
                                             unsigned int index,
                                             int flags);

void vftr_register_ompt_callback_implicit_task(ompt_set_callback_t ompt_set_callback);

#endif
