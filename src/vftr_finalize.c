#include "self_profile.h"
#include "off_hooks.h"
#include "cyghooks.h"
#include "vftr_hooks.h"
#include "vftrace_state.h"
#include "environment.h"
#include "symbols.h"
#include "processes.h"
#include "stacks.h"
#include "threadstacks.h"
#include "collate_ranks.h"
#include "logfile.h"
#include "ranklogfile.h"
#include "sampling.h"
#include "timer.h"

void vftr_finalize() {
   if (vftrace.state == off || vftrace.state == uninitialized) {
      // was already finalized
      // Maybe by MPI_Finalize
      // vftr_finalize was already registered by atexit
      // before vftrace knew that this was an MPI-program
      return;
   }
   SELF_PROFILE_START_FUNCTION;

   // update the vftrace state
   vftrace.state = off;
   // set end timer string
   long long int runtime = vftr_get_runtime_nsec();
   vftrace.timestrings.end_time = vftr_get_date_str();

   // in case finalize was not called from the threadstacks root
   // the threadstack needs to be poped completely
   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);
   while (my_threadstack->stackID > 0) {
      vftr_function_exit(NULL, NULL);
      my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
      my_threadstack = vftr_get_my_threadstack(my_thread);
   }


   // finalize stacks
   vftr_finalize_stacktree(&(vftrace.process.stacktree));

   // collate stacks and other information
   // among all processes
   // to get a global ordering of stacks
   vftr_collate_ranks(&vftrace);

   // write summary logfile for all ranks
   vftr_write_logfile(vftrace, runtime);
   // write logfile for individual ranks
   vftr_write_ranklogfile(vftrace, runtime);

   // finish sampling
   // add the sampling of leaving init to the vfd-file
   vftr_sample_init_function_exit(&(vftrace.sampling), runtime);
   vftr_finalize_sampling(&(vftrace.sampling), vftrace.environment,
                          vftrace.process, vftrace.timestrings,
                          (double) (runtime * 1.0e-9));

   // free the dynamic process data
   vftr_process_free(&vftrace.process);

   // free the symbol table
   vftr_symboltable_free(&vftrace.symboltable);

   // free the environment to avoid memory leaks
   vftr_environment_free(&(vftrace.environment));

   // free the timer strings
   vftr_timestrings_free(&(vftrace.timestrings));

   // redirect the function entry and exit hooks to deactivate them
   // use a dummy function that does nothing
   vftr_set_enter_func_hook(vftr_function_hook_off);
   vftr_set_exit_func_hook(vftr_function_hook_off);

   SELF_PROFILE_END_FUNCTION;
   FINALIZE_SELF_PROF_VFTRACE;
}
