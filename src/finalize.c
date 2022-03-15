#ifdef _DEBUG
#include <stdio.h>
#endif

#include "off_hooks.h"
#include "cyghooks.h"
#include "vftr_hooks.h"
#include "vftrace_state.h"
#include "environment.h"
#include "symbols.h"
#include "processes.h"
#include "stacks.h"
#include "logfile.h"
#include "timer.h"

void vftr_finalize() {
   // update the vftrace state
   vftrace.state = finalized;
   // set end timer string
   vftrace.timestrings.end_time = vftr_get_date_str();
#ifdef _DEBUG
   fprintf(stderr, "Vftrace finalized at ");
   vftr_print_date_str(stderr);
   fprintf(stderr, "\n");
#endif

   // write logfile
   vftr_write_logfile(vftrace);



   // free the dynamic process data
#ifdef _DEBUG
   vftr_print_stacktree(stderr, vftrace.process.stacktree);
#endif
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
