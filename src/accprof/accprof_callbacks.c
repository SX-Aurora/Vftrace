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

// Each OpenACC region is identified by its event type, the source file lines and the stack ID
// of the parent stack (the new stack is not yet created at this point). The latter is relevant
// for the case in which two different OpenACC regions are called with the same source file lines.
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

typedef struct open_wait {
   long long start_time;
   int async;
   stack_t *stack;
   struct open_wait *next;
} open_wait_t;

open_wait_t *open_wait_queues = NULL;
static int n_open_queues = 0;

void vftr_append_wait_queue (long long start_time, int async, stack_t *stack) {
   if (open_wait_queues == NULL) {
	open_wait_queues = (open_wait_t*)malloc(sizeof(open_wait_t));
        open_wait_queues->start_time = start_time;
        open_wait_queues->async = async;
        open_wait_queues->stack = stack;
        open_wait_queues->next = NULL;
   } else {
	open_wait_t *this_queue = open_wait_queues;
        while (this_queue->next != NULL) {
            this_queue = this_queue->next;
        }
        this_queue->next = (open_wait_t*)malloc(sizeof(open_wait_t));
     	this_queue = this_queue->next;
	this_queue->start_time = start_time;
        this_queue->async = async;
  	this_queue->stack = stack;
	this_queue->next = NULL; 
   }
   n_open_queues++;
}

open_wait_t *vftr_pick_wait_queue (int async) {
	open_wait_t *this_queue = open_wait_queues;
        open_wait_t *prev_queue = NULL;
	for (int i = 0; i < n_open_queues; i++) {
	   if (this_queue->async == async) {
		if (prev_queue == NULL) { // First list element
		   open_wait_queues = this_queue->next;
                } else if (this_queue->next == NULL) { // Last element
		   prev_queue->next = NULL;
                } else { // Inbetween
                   prev_queue->next = this_queue->next;
                }
 		n_open_queues--;
		return this_queue;
           }
           prev_queue = this_queue;
 	   this_queue = this_queue->next;
        }
	// No match found
	return NULL;
}

// The hook for acc_ev_wait_start.
// It is similar to the hooks for the other event classes, except that the new stack is popped again at the end. This way, if another OpenACC call happens during the wait which is not the corresponding
// acc_ev_wait_end, it is not considered to be called by the wait call.
// Additionally, the region identifier, the start time and the current stack are stored in a list of open wait event queues. When the matching acc_ev_wait_end happens, it computes the difference between its timestamp and this one, and stores it into the profile obtained from the stack of this region.
void prof_wait_start (acc_prof_info *prof_info, acc_event_info *event_info, acc_api_info *api_info) {
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


   vftr_append_wait_queue (vftr_get_runtime_nsec(), prof_info->async_queue, my_new_stack);

   vftr_accumulate_accprofiling (&(my_profile->accprof), prof_info->event_type,
                                 prof_info->line_no, prof_info->end_line_no,
                                 prof_info->src_file, prof_info->func_name,
                                 NULL, NULL, 0);

   threadstacklist_t stacklist = my_thread->stacklist;
   (void)vftr_threadstack_pop(&(my_thread->stacklist));
   
   vftr_accumulate_accprofiling_overhead (&(my_profile->accprof),
                                          vftr_get_runtime_nsec() - wait_begin_time);
}

// As explained above, the hook for acc_ev_wait_start immediately pops the newly created stack.
// Therefore, in contrast to the default exit hook, this is omitted here.
// We search the list of open queues for the matching queue. Using the timestamp stored there,
// we accumulate the callprofiling.
// TODO: What if no matching queue is found?
void prof_wait_end (acc_prof_info *prof_info, acc_event_info *event_info, acc_api_info *api_info) {
   stack_t *stack = NULL;
   long long wait_end_time = vftr_get_runtime_nsec();
   open_wait_t *match_queue = vftr_pick_wait_queue (prof_info->async_queue);
   
   if (match_queue != NULL) {
      thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
      profile_t *my_profile = vftr_get_my_profile (match_queue->stack, my_thread);
      vftr_accumulate_callprofiling (&(my_profile->callprof), 1,
                                     wait_end_time - match_queue->start_time);
      vftr_accumulate_accprofiling_overhead (&(my_profile->accprof),
                                             vftr_get_runtime_nsec() - wait_end_time);
      free(match_queue);
   }
}

// The hook for OpenACC start events (except acc_ev_wait_start).
// The stack structure corresponds to the stack structure
// of the default function hooks, so that OpenACC regions are included in the stack tree.
// Additionally, we accumulate the acc profiles depending on whether the event
// is a launch, data or other event.
// Some events do not have matching _end events (create, delete, alloc, free). It is therefore
// not possible to assign a start and end time to them. For these events, the argument
// acc_callprof needs to be set to false, so that no callprofiling its accumulated.
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

// The hook for OpenACC end events (except acc_ev_wait_end).
// Analogous to vftr_accprof_region_begin.
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

// Wrappers for different kind of hooks.
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

// Some events do not have matching _start and _end events (e.g. create, delete, alloc, free).
// We do not measure a runtime for them, wherefore the last argument is set to false.
// By entering and exiting the region immediately we make a stack entry for them.
void prof_single (acc_prof_info *prof_info, acc_event_info *event_info, acc_api_info *api_info) {
   vftr_accprof_region_begin (prof_info, event_info, false);
   vftr_accprof_region_end (prof_info, event_info, false);
}

// Function pointers to registration routines. Are set by acc_register_library.
static acc_prof_reg vftr_accprof_register = NULL;
static acc_prof_reg vftr_accprof_unregister = NULL;

// This is the main OpenACC runtime hook. DO NOT CHANGE ITS NAME. The OpenACC searches for
// this symbol and executes it, providing the necessary arguments.
// This function is evoked regardless if Vftrace is switched off or not.
// Therefore, we need to return immediately when acc_register_library is called from
// within a deactivated Vftrace. In that case, vetco_callback_registration is set to true.
void acc_register_library (acc_prof_reg register_ev, acc_prof_reg unregister_ev,
                           acc_prof_lookup_func lookup) {
   if (vftrace.accprof_state.veto_callback_registration) return;
   vftr_accprof_register = register_ev;
   vftr_accprof_unregister = unregister_ev;
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
   vftr_accprof_register (acc_ev_wait_start, prof_wait_start, 0);
   vftr_accprof_register (acc_ev_wait_end, prof_wait_end, 0);
}

void vftr_unregister_accprof_callbacks () {
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
   vftr_accprof_unregister (acc_ev_wait_start, prof_wait_start, 0);
   vftr_accprof_unregister (acc_ev_wait_end, prof_wait_end, 0);
}

// In some cases, OpenACC is already setup before vftr_initialize is called.
// If so, we need to unregister the callbacks, so that they are not called from within
// an inactive Vftrace.
void vftr_veto_accprof_callbacks () {
   if (vftr_accprof_unregister != NULL) vftr_unregister_accprof_callbacks();
   vftrace.accprof_state.veto_callback_registration = true;
}

