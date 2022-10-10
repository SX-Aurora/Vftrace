#include <stdio.h>

#include <omp.h>
#include <omp-tools.h>

#include "omp_regions.h"

static void vftr_ompt_callback_parallel_begin(ompt_data_t *encountering_task_data,
                                              const ompt_frame_t *encountering_task_frame,
                                              ompt_data_t *parallel_data,
                                              unsigned int requested_parallelism,
                                              int flags, const void *codeptr_ra) {
   (void) encountering_task_data;
   (void) encountering_task_frame;
   (void) parallel_data;
   (void) requested_parallelism;
   (void) flags;
   vftr_omp_region_begin("omp_parallel_region", codeptr_ra);
}

void vftr_register_ompt_callback_parallel_begin(ompt_set_callback_t ompt_set_callback) {
   ompt_callback_parallel_begin_t f_ompt_callback_parallel_begin =
      &vftr_ompt_callback_parallel_begin;
   int ompt_set = ompt_set_callback(ompt_callback_parallel_begin,
                                  (ompt_callback_t)f_ompt_callback_parallel_begin);
   if (ompt_set == ompt_set_never) {
      fprintf(stderr, "0: Could not register callback \"ompt_callback_parallel_begin\"\n");
   }
}
