#include <stdlib.h>

#ifdef _OMP
#include "start_tool.h"
#endif

#include "timer.h"
#include "off_hooks.h"
#include "cyghooks.h"
#include "vftr_hooks.h"
#include "vftrace_state.h"
#include "environment.h"
#include "symbols.h"
#include "vftr_finalize.h"
#include "processes.h"
#include "sampling.h"

void vftr_initialize(void *func, void *call_site) {
   // First step is to initialize the reference timer
   vftr_set_local_ref_time();

   // parse the relevant environment variables
   vftrace.environment = vftr_read_environment();

   if (vftrace.environment.vftrace_off.value.bool_val) {
      // update the vftrace state
      vftrace.state = off;
      // free the environment to avoid memory leaks
      vftr_environment_free(&(vftrace.environment));
      // set the function hooks to a dummy function that does nothing
      vftr_set_enter_func_hook(vftr_function_hook_off);
      vftr_set_exit_func_hook(vftr_function_hook_off);
   } else {
      // update the vftrace state
      vftrace.state = on;

      // set start time string
      vftrace.timestrings.start_time = vftr_get_date_str();

      // read the symbol table of the executable and its libraries
      vftrace.symboltable = vftr_read_symbols();
      vftr_symboltable_determine_preciseness(&(vftrace.symboltable),
                                             vftrace.environment.preciseregex.value.regex_val);

      // initialize the dynamic process data
      vftrace.process = vftr_new_process();

      // initialize possible sampling
      vftrace.sampling = vftr_new_sampling(vftrace.environment);

      // assign the appropriate function hooks to handle sampling.
      vftr_set_enter_func_hook(vftr_function_entry);
      vftr_set_exit_func_hook(vftr_function_exit);

      // trick the linker into including the omp callback symbols
#ifdef _OMP
      (void) ompt_start_tool(0, NULL);
#endif

      // set the finalize function to be executed at the termination of the program
      atexit(vftr_finalize);

      // now that initializing is done the actual hook needs
      // to be called with the appropriate arguments
      vftr_function_entry(func, call_site);
   }
}
