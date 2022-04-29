#include "off_hooks.h"
#include "cyghooks.h"
#include "vftr_hooks.h"
#include "vftrace_state.h"
#include "environment.h"
#include "symbols.h"
#include "processes.h"
#include "stacks.h"
#include "collate_stacks.h"
#include "logfile.h"
#include "sampling.h"
#include "timer.h"

void vftr_finalize() {
   // update the vftrace state
   vftrace.state = off;
   // set end timer string
   long long int runtime = vftr_get_runtime_usec();
   vftrace.timestrings.end_time = vftr_get_date_str();

   // finalize stacks
   vftr_finalize_stacktree(&(vftrace.process.stacktree));

   // collate stacks among all processes
   // to get a global ordering of stacks
   vftrace.process.collated_stacktree =
      vftr_collate_stacks(&(vftrace.process.stacktree));

   // write logfile
   vftr_write_logfile(vftrace, runtime);

   // finish sampling
   vftr_finalize_sampling(&(vftrace.sampling), vftrace.environment,
                          vftrace.process, vftrace.timestrings,
                          (double) (runtime * 1.0e-6));





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
}
