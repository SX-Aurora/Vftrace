#include <stdio.h>
#include <string.h>

#include "thread_types.h"
#include "threadstack_types.h"
#include "stack_types.h"
#include "vftrace_state.h"
#include "hashing.h"
#include "threads.h"
#include "threadstacks.h"


#include "acc_prof.h"

void vftr_accprof_region_begin (acc_prof_info *prof_info, acc_event_info *event_info) {
   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);
   stack_t *my_stack = vftrace.process.stacktree.stacks + my_threadstack->stackID;

   //char *func_name = prof_info->func_name; 
   //uint64_t pseudo_addr = vftr_jenkins_murmur_64_hash (strlen(func_name), (uint8_t*)func_name);
   //pseudo_addr += prof_info->line_no + prof_info->end_line_no; 

   //my_threadstack = vftr_update_threadstack_region (my_threadstack, my_thread,
   //                                                 (uintptr_t)pseudo_addr, func_name,
   //                                                 &vftrace, false);
   //stack_t *my_new_stack = vftrace.process.stacktree.stacks + my_threadstack->stackID;
}

void vftr_accprof_region_end (acc_prof_info *prof_info, acc_event_info *event_info) {
   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);
   stack_t *my_stack = vftrace.process.stacktree.stacks + my_threadstack->stackID;

   //(void)vftr_threadstack_pop(&(my_thread->stacklist));
}

void prof_data_start (acc_prof_info *prof_info, acc_event_info *event_info, acc_api_info *api_info) {
   vftr_accprof_region_begin (prof_info, event_info);
}

void prof_data_end (acc_prof_info *prof_info, acc_event_info *event_info, acc_api_info *api_info) {
   vftr_accprof_region_end (prof_info, event_info); 
}

void prof_launch_start (acc_prof_info *prof_info, acc_event_info *event_info, acc_api_info *api_info) {
   printf ("Start region\n");
   vftr_accprof_region_begin (prof_info, event_info);
}

void prof_launch_end (acc_prof_info *prof_info, acc_event_info *event_info, acc_api_info *api_info) {
   printf ("End region\n");
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
