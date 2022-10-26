#include <stdio.h>
#include <string.h>

#include "thread_types.h"
#include "threadstack_types.h"
#include "stack_types.h"
#include "callprofiling.h"
#include "vftrace_state.h"
#include "hashing.h"
#include "threads.h"
#include "threadstacks.h"
#include "misc_utils.h"


#include "acc_prof.h"
#include "accprofiling.h"

char *concatenate_openacc_name (acc_event_t event_type, int line_1, int line_2) {
   int n1 = vftr_count_base_digits ((long long)line_1, 10) + 1;
   int n2 = vftr_count_base_digits ((long long)line_2, 10) + 1;
   int n3 = vftr_count_base_digits ((long long)event_type, 10) + 1;
   int new_len = strlen("openacc") + n1 + n2 + n3 + 1; 
   char *s = (char*)malloc(new_len * sizeof(char));
   snprintf (s, new_len, "openacc_%d_%d_%d\n", line_1, line_2, event_type);
   return s;
}

void vftr_accprof_region_begin (acc_prof_info *prof_info, acc_event_info *event_info) {

   long long region_begin_time_begin = vftr_get_runtime_nsec();

   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);
   stack_t *my_stack = vftrace.process.stacktree.stacks + my_threadstack->stackID;

   char *pseudo_name = concatenate_openacc_name (prof_info->event_type,
                                                 prof_info->line_no, prof_info->end_line_no);
   uint64_t pseudo_addr = vftr_jenkins_murmur_64_hash (strlen(pseudo_name), (uint8_t*)pseudo_name);

   my_threadstack = vftr_update_threadstack_region (my_threadstack, my_thread,
                                                    (uintptr_t)pseudo_addr, pseudo_name,
                                                    &vftrace, false);
   stack_t *my_new_stack = vftrace.process.stacktree.stacks + my_threadstack->stackID;
   profile_t *my_profile = vftr_get_my_profile(my_new_stack, my_thread);
   vftr_accumulate_callprofiling(&(my_profile->callprof), 1, -region_begin_time_begin);

   acc_launch_event_info *launch_event_info;
   acc_data_event_info *data_event_info;
   switch (prof_info->event_type) {
      case acc_ev_enqueue_launch_start:
      case acc_ev_enqueue_launch_end:
	launch_event_info = (acc_launch_event_info*)event_info;
        vftr_accumulate_accprofiling (&(my_profile->accprof), prof_info->event_type,
				      prof_info->line_no, prof_info->end_line_no,
                                      prof_info->src_file,
                                      launch_event_info->kernel_name, NULL, 0);
        break;
      case acc_ev_enqueue_upload_start:
      case acc_ev_enqueue_upload_end:
      case acc_ev_enqueue_download_start:
      case acc_ev_enqueue_download_end:
         data_event_info = (acc_data_event_info*)event_info;
         vftr_accumulate_accprofiling (&(my_profile->accprof), prof_info->event_type,
				       prof_info->line_no, prof_info->end_line_no,
                                       prof_info->src_file,
                                       NULL, data_event_info->var_name, data_event_info->bytes);
         break;
      default:
         vftr_accumulate_accprofiling (&(my_profile->accprof), prof_info->event_type,
				       prof_info->line_no, prof_info->end_line_no,
                                       prof_info->src_file,
                                       NULL, NULL, 0);
    }
}

void vftr_accprof_region_end (acc_prof_info *prof_info, acc_event_info *event_info) {

   long long region_end_time_begin = vftr_get_runtime_nsec();

   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);
   stack_t *my_stack = vftrace.process.stacktree.stacks + my_threadstack->stackID;
   profile_t *my_profile = vftr_get_my_profile(my_stack, my_thread);
   vftr_accumulate_callprofiling(&(my_profile->callprof), 0, region_end_time_begin);

   threadstacklist_t stacklist = my_thread->stacklist;
   (void)vftr_threadstack_pop(&(my_thread->stacklist));

}

void prof_data_start (acc_prof_info *prof_info, acc_event_info *event_info, acc_api_info *api_info) {
   vftr_accprof_region_begin (prof_info, event_info);
}

void prof_data_end (acc_prof_info *prof_info, acc_event_info *event_info, acc_api_info *api_info) {
   vftr_accprof_region_end (prof_info, event_info); 
}

void prof_launch_start (acc_prof_info *prof_info, acc_event_info *event_info, acc_api_info *api_info) {
   vftr_accprof_region_begin (prof_info, event_info);
}

void prof_launch_end (acc_prof_info *prof_info, acc_event_info *event_info, acc_api_info *api_info) {
   vftr_accprof_region_end (prof_info, event_info);
}

void acc_register_library (acc_prof_reg register_ev, acc_prof_reg unregister_ev,
                           acc_prof_lookup_func lookup) {
   printf ("Register library!\n");

   register_ev (acc_ev_enqueue_upload_start, prof_data_start, 0);
   register_ev (acc_ev_enqueue_upload_end, prof_data_end, 0);
   register_ev (acc_ev_enqueue_download_start, prof_data_start, 0);
   register_ev (acc_ev_enqueue_download_end, prof_data_end, 0);
   register_ev (acc_ev_enqueue_launch_start, prof_launch_start, 0);
   register_ev (acc_ev_enqueue_launch_end, prof_launch_end, 0);

}
