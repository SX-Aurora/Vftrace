#include <stdio.h>

#include <omp.h>
#include <omp-tools.h>

static void vftr_ompt_callback_parallel_end(ompt_data_t *parallel_data,
                                            ompt_data_t *encountering_task_data,
                                            int flags, const void *codeptr_ra) {
   fprintf(stderr, "ending parallel region\n");
}

void vftr_register_ompt_callback_parallel_end(ompt_set_callback_t ompt_set_callback) {
   ompt_callback_parallel_end_t f_ompt_callback_parallel_end =
      &vftr_ompt_callback_parallel_end;
   int ompt_set = ompt_set_callback(ompt_callback_parallel_end,
                                    (ompt_callback_t)f_ompt_callback_parallel_end);
   if (ompt_set == ompt_set_never) {
      fprintf(stderr, "0: Could not register callback \"ompt_callback_parallel_end\"\n");
   }
}
