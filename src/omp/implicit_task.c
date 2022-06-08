#include <stdio.h>

#include <omp.h>
#include <omp-tools.h>

#include "threads.h"
#include "vftrace_state.h"

static void vftr_ompt_callback_implicit_task(ompt_scope_endpoint_t endpoint,
                                             ompt_data_t *parallel_data,
                                             ompt_data_t *task_data,
                                             unsigned int actual_parallelism,
                                             unsigned int index,
                                             int flags) {
//   switch (endpoint) {
//      case ompt_scope_begin:
//         //thread_t *my_thread = vftr_get_my_thread(&vftrace.process.threadtree);
//
//         fprintf(stderr, "implicit_task begin:  t=%d l=%d\n", omp_get_thread_num(), omp_get_level());
//         break;
//      case ompt_scope_end:
//         fprintf(stderr, "implicit_task end:    t=%d l=%d\n", omp_get_thread_num(), omp_get_level());
//         break;
//      case ompt_scope_beginend:
//         printf("implicit_task begend: t=%d\n", omp_get_thread_num());
//         break;
//      default:
//         break;
//   }
}

void vftr_register_ompt_callback_implicit_task(ompt_set_callback_t ompt_set_callback) {
   ompt_callback_implicit_task_t f_ompt_callback_implicit_task =
      &vftr_ompt_callback_implicit_task;
   int ompt_set =  ompt_set_callback(ompt_callback_implicit_task,
                                     (ompt_callback_t)f_ompt_callback_implicit_task);
   if (ompt_set == ompt_set_never) {
      fprintf(stderr, "0: Could not register callback \"ompt_callback_implicit_task\"\n");
   }
}
