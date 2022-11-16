#include <stdio.h>
#include <string.h>

#include "thread_types.h"
#include "threadstack_types.h"
#include "stack_types.h"
#include "profiling.h"
#include "callprofiling.h"
#include "vftrace_state.h"
#include "hashing.h"
#include "threads.h"
#include "threadstacks.h"
#include "misc_utils.h"
#include "timer.h"


#include "acc_prof.h"
#include "accprofiling.h"

char *concatenate_openacc_name (acc_event_t event_type, int line_1, int line_2, int parent_id) {
   int n1 = vftr_count_base_digits ((long long)line_1, 10) + 1;
   int n2 = vftr_count_base_digits ((long long)line_2, 10) + 1;
   int n3 = vftr_count_base_digits ((long long)event_type, 10) + 1;
   int n4 = vftr_count_base_digits ((long long)parent_id, 10) + 1;
   int new_len = strlen("openacc") + n1 + n2 + n3 + n4 + 1; 
   char *s = (char*)malloc(new_len * sizeof(char));
   snprintf (s, new_len, "openacc_%d_%d_%d_%d", line_1, line_2, event_type, parent_id);
   return s;
}

void vftr_accprof_region_begin (acc_prof_info *prof_info, acc_event_info *event_info,
                                bool acc_callprof) {

   long long region_begin_time_begin = vftr_get_runtime_nsec();

   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);
   stack_t *my_stack = vftrace.process.stacktree.stacks + my_threadstack->stackID;

   char *pseudo_name = concatenate_openacc_name (prof_info->event_type,
                                                 prof_info->line_no, prof_info->end_line_no,
						 my_stack->lid);
   uint64_t pseudo_addr = vftr_jenkins_murmur_64_hash (strlen(pseudo_name), (uint8_t*)pseudo_name);

   my_threadstack = vftr_update_threadstack_region (my_threadstack, my_thread,
                                                    (uintptr_t)pseudo_addr, pseudo_name,
                                                    &vftrace, false);
   stack_t *my_new_stack = vftrace.process.stacktree.stacks + my_threadstack->stackID;
   profile_t *my_profile = vftr_get_my_profile(my_new_stack, my_thread);
   if (acc_callprof) vftr_accumulate_callprofiling(&(my_profile->callprof), 1, -region_begin_time_begin);

   acc_launch_event_info *launch_event_info;
   acc_data_event_info *data_event_info;
   switch (prof_info->event_type) {
      case acc_ev_enqueue_launch_start:
      case acc_ev_enqueue_launch_end:
        launch_event_info = (acc_launch_event_info*)event_info;
        vftr_accumulate_accprofiling (&(my_profile->accprof), prof_info->event_type,
        			      prof_info->line_no, prof_info->end_line_no,
                                      prof_info->src_file, prof_info->func_name,
                                      launch_event_info->kernel_name, NULL, 0);
        break;
      case acc_ev_enqueue_upload_start:
      case acc_ev_enqueue_upload_end:
      case acc_ev_enqueue_download_start:
      case acc_ev_enqueue_download_end:
      case acc_ev_create:
      case acc_ev_delete:
      case acc_ev_alloc:
      case acc_ev_free:
         data_event_info = (acc_data_event_info*)event_info;
         vftr_accumulate_accprofiling (&(my_profile->accprof), prof_info->event_type,
        			       prof_info->line_no, prof_info->end_line_no,
                                       prof_info->src_file, prof_info->func_name,
                                       NULL, data_event_info->var_name, data_event_info->bytes);
         break;
      default:
         vftr_accumulate_accprofiling (&(my_profile->accprof), prof_info->event_type,
        			       prof_info->line_no, prof_info->end_line_no,
                                       prof_info->src_file, prof_info->func_name,
                                       NULL, NULL, 0);
   }

   vftr_accumulate_accprofiling_overhead (&(my_profile->accprof),
                                          vftr_get_runtime_nsec() - region_begin_time_begin);
}

void vftr_accprof_region_end (acc_prof_info *prof_info, acc_event_info *event_info,
                              bool acc_callprof) {

   long long region_end_time_begin = vftr_get_runtime_nsec();

   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);
   stack_t *my_stack = vftrace.process.stacktree.stacks + my_threadstack->stackID;
   profile_t *my_profile = vftr_get_my_profile(my_stack, my_thread);
   if (acc_callprof) vftr_accumulate_callprofiling(&(my_profile->callprof), 0, region_end_time_begin);

   threadstacklist_t stacklist = my_thread->stacklist;
   (void)vftr_threadstack_pop(&(my_thread->stacklist));

   vftr_accumulate_accprofiling_overhead (&(my_profile->accprof),
                                          vftr_get_runtime_nsec() - region_end_time_begin);

}

void prof_data_start (acc_prof_info *prof_info, acc_event_info *event_info, acc_api_info *api_info) {
   vftr_accprof_region_begin (prof_info, event_info, true);
}

void prof_data_end (acc_prof_info *prof_info, acc_event_info *event_info, acc_api_info *api_info) {
   vftr_accprof_region_end (prof_info, event_info, true); 
}

void prof_compute_start (acc_prof_info *prof_info, acc_event_info *event_info, acc_api_info *api_info) {
   vftr_accprof_region_begin (prof_info, event_info, true);
}

void prof_compute_end (acc_prof_info *prof_info, acc_event_info *event_info, acc_api_info *api_info) {
   vftr_accprof_region_end (prof_info, event_info, true);
}

void prof_single (acc_prof_info *prof_info, acc_event_info *event_info, acc_api_info *api_info) {
   vftr_accprof_region_begin (prof_info, event_info, false);
   vftr_accprof_region_end (prof_info, event_info, false);
}

struct open_wait {
   long long start_time;
   stack_t *stack;
   int async;
};

#define MAX_QUEUES 100
struct open_wait open_queues[MAX_QUEUES];
int n_open_queues = 0;

void prof_wait_start (acc_prof_info *prof_info, acc_event_info *event_info, acc_api_info *api_info) {
   //stack_t *stack = vftr_accprof_region_begin (prof_info, event_info, false); 
   long long wait_begin_time = vftr_get_runtime_nsec();

   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);
   stack_t *my_stack = vftrace.process.stacktree.stacks + my_threadstack->stackID;

   char *pseudo_name = concatenate_openacc_name (prof_info->event_type,
                                                 prof_info->line_no, prof_info->end_line_no,
						 my_stack->lid);
   uint64_t pseudo_addr = vftr_jenkins_murmur_64_hash (strlen(pseudo_name), (uint8_t*)pseudo_name);

   my_threadstack = vftr_update_threadstack_region (my_threadstack, my_thread,
                                                    (uintptr_t)pseudo_addr, pseudo_name,
                                                    &vftrace, false);
   stack_t *my_new_stack = vftrace.process.stacktree.stacks + my_threadstack->stackID;
   profile_t *my_profile = vftr_get_my_profile(my_new_stack, my_thread);


   open_queues[n_open_queues].start_time = vftr_get_runtime_nsec();
   open_queues[n_open_queues].stack = my_new_stack;
   open_queues[n_open_queues].async = prof_info->async_queue;
   n_open_queues++;

   vftr_accumulate_accprofiling (&(my_profile->accprof), prof_info->event_type,
                                 prof_info->line_no, prof_info->end_line_no,
                                 prof_info->src_file, prof_info->func_name,
                                 NULL, NULL, 0);

   threadstacklist_t stacklist = my_thread->stacklist;
   (void)vftr_threadstack_pop(&(my_thread->stacklist));
   
   vftr_accumulate_accprofiling_overhead (&(my_profile->accprof),
                                          vftr_get_runtime_nsec() - wait_begin_time);
}

void prof_wait_end (acc_prof_info *prof_info, acc_event_info *event_info, acc_api_info *api_info) {
   stack_t *stack = NULL;
   long long wait_end_time = vftr_get_runtime_nsec();
   long long wait_begin_time;
   for (int i = 0; i < n_open_queues; i++) {
      if (prof_info->async_queue == open_queues[i].async) {
   	 stack = open_queues[i].stack;
         wait_begin_time = open_queues[i].start_time;
         n_open_queues--;
         break;
      }
   }

   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   profile_t *my_profile = vftr_get_my_profile (stack, my_thread);
   vftr_accumulate_callprofiling (&(my_profile->callprof), 1, wait_end_time - wait_begin_time);
   vftr_accumulate_accprofiling_overhead (&(my_profile->accprof),
                                          vftr_get_runtime_nsec() - wait_end_time);
}

void prof_dummy (acc_prof_info *prof_info, acc_event_info *event_info, acc_api_info *api_info) {
   printf ("Dummy region!\n");
}

static acc_prof_reg vftr_accprof_register = NULL;
static acc_prof_reg vftr_accprof_unregister = NULL;

void acc_register_library (acc_prof_reg register_ev, acc_prof_reg unregister_ev,
                           acc_prof_lookup_func lookup) {
   if (vftrace.accprof_state.veto_callback_registration) return;
   vftr_accprof_register = register_ev;
   vftr_accprof_unregister = unregister_ev;
   /// Problems with creating stack entries for these event types
   ///vftr_accprof_register (acc_ev_device_init_start, prof_data_start, 0);
   ///vftr_accprof_register (acc_ev_device_init_end, prof_data_end, 0);
   ///vftr_accprof_register (acc_ev_device_shutdown_start, prof_data_start, 0);
   ///vftr_accprof_register (acc_ev_device_shutdown_end, prof_data_end, 0);
   ///vftr_accprof_register (acc_ev_runtime_shutdown, prof_single, 0);
   vftr_accprof_register (acc_ev_create, prof_single, 0);
   vftr_accprof_register (acc_ev_delete, prof_single, 0);
   vftr_accprof_register (acc_ev_alloc, prof_single, 0);
   vftr_accprof_register (acc_ev_free, prof_single, 0);
   vftr_accprof_register (acc_ev_enter_data_start, prof_compute_start, 0);
   vftr_accprof_register (acc_ev_enter_data_end, prof_compute_end, 0);
   vftr_accprof_register (acc_ev_exit_data_start, prof_compute_start, 0);
   vftr_accprof_register (acc_ev_exit_data_end, prof_compute_end, 0);
   vftr_accprof_register (acc_ev_update_start, prof_data_start, 0); 
   vftr_accprof_register (acc_ev_update_end, prof_data_end, 0); 
   vftr_accprof_register (acc_ev_compute_construct_start, prof_compute_start, 0);
   vftr_accprof_register (acc_ev_compute_construct_end, prof_compute_end, 0);
   vftr_accprof_register (acc_ev_enqueue_launch_start, prof_compute_start, 0);
   vftr_accprof_register (acc_ev_enqueue_launch_end, prof_compute_end, 0);
   vftr_accprof_register (acc_ev_enqueue_upload_start, prof_data_start, 0);
   vftr_accprof_register (acc_ev_enqueue_upload_end, prof_data_end, 0);
   vftr_accprof_register (acc_ev_enqueue_download_start, prof_data_start, 0);
   vftr_accprof_register (acc_ev_enqueue_download_end, prof_data_end, 0);
   /// Asynchronous events are not yet supported.
   vftr_accprof_register (acc_ev_wait_start, prof_wait_start, 0);
   vftr_accprof_register (acc_ev_wait_end, prof_wait_end, 0);
}

void vftr_unregister_accprof_callbacks () {
   ///vftr_accprof_unregister (acc_ev_device_init_start, prof_data_start, 0);
   ///vftr_accprof_unregister (acc_ev_device_init_end, prof_data_end, 0);
   ///vftr_accprof_unregister (acc_ev_device_shutdown_start, prof_data_start, 0);
   ///vftr_accprof_unregister (acc_ev_device_shutdown_end, prof_data_end, 0);
   ///vftr_accprof_unregister (acc_ev_runtime_shutdown, prof_single, 0);
   vftr_accprof_unregister (acc_ev_create, prof_single, 0);
   vftr_accprof_unregister (acc_ev_delete, prof_single, 0);
   vftr_accprof_unregister (acc_ev_alloc, prof_single, 0);
   vftr_accprof_unregister (acc_ev_free, prof_single, 0);
   vftr_accprof_unregister (acc_ev_enter_data_start, prof_compute_start, 0);
   vftr_accprof_unregister (acc_ev_enter_data_end, prof_compute_end, 0);
   vftr_accprof_unregister (acc_ev_exit_data_start, prof_compute_start, 0);
   vftr_accprof_unregister (acc_ev_exit_data_end, prof_compute_end, 0);
   vftr_accprof_unregister (acc_ev_update_start, prof_data_start, 0); 
   vftr_accprof_unregister (acc_ev_update_end, prof_data_end, 0); 
   vftr_accprof_unregister (acc_ev_compute_construct_start, prof_compute_start, 0);
   vftr_accprof_unregister (acc_ev_compute_construct_end, prof_compute_end, 0);
   vftr_accprof_unregister (acc_ev_enqueue_launch_start, prof_compute_start, 0);
   vftr_accprof_unregister (acc_ev_enqueue_launch_end, prof_compute_end, 0);
   vftr_accprof_unregister (acc_ev_enqueue_upload_start, prof_data_start, 0);
   vftr_accprof_unregister (acc_ev_enqueue_upload_end, prof_data_end, 0);
   vftr_accprof_unregister (acc_ev_enqueue_download_start, prof_data_start, 0);
   vftr_accprof_unregister (acc_ev_enqueue_download_end, prof_data_end, 0);
   ///vftr_accprof_unregister (acc_ev_wait_start, prof_data_start, 0);
   ///vftr_accprof_unregister (acc_ev_wait_end, prof_data_end, 0);
}

void vftr_veto_accprof_callbacks () {
   if (vftr_accprof_unregister != NULL) vftr_unregister_accprof_callbacks();
   vftrace.accprof_state.veto_callback_registration = true;
}

