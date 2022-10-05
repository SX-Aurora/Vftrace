#include <stdlib.h>

#ifdef _OMP
#include "start_tool.h"
#endif

#ifdef _CUPTI
#include "cupti.h"
#endif

#include "self_profile.h"
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
   INIT_SELF_PROF_VFTRACE;
   SELF_PROFILE_START_FUNCTION;
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
      SELF_PROFILE_END_FUNCTION;
      FINALIZE_SELF_PROF_VFTRACE;
   } else {
      // update the vftrace state
      vftrace.state = on;

      // set start time string
      vftrace.timestrings.start_time = vftr_get_date_str();

      // check environment sanity
      vftr_environment_assert(stderr, vftrace.environment);
      vftr_check_env_names(stderr, &(vftrace.environment));

      // read the symbol table of the executable and its libraries
      vftrace.symboltable = vftr_read_symbols();
      vftr_symboltable_determine_preciseness(&(vftrace.symboltable),
                                             vftrace.environment.preciseregex.value.regex_val);
      vftr_symboltable_strip_fortran_module_name(&(vftrace.symboltable),
                                                 vftrace.environment.strip_module_names.value.bool_val);
#ifdef _LIBERTY
      vftr_symboltable_demangle_cxx_name(&(vftrace.symboltable),
                                         vftrace.environment.demangle_cxx.value.bool_val);
#endif


      // initialize the dynamic process data
      vftrace.process = vftr_new_process();

      // initialize possible sampling
      vftrace.sampling = vftr_new_sampling(vftrace.environment);

      // assign the appropriate function hooks to handle sampling.
      vftr_set_enter_func_hook(vftr_function_entry);
      vftr_set_exit_func_hook(vftr_function_exit);

      // trick the linker into including extra symbols
#ifdef _OMP
      // omp callback symbols
      (void) ompt_start_tool(0, NULL);
#endif

#ifdef _CUPTI
      if (cupti_initialize() != 0) {
	printf ("No devices found. No GPU profile will be generated\n");
      }
#endif

      // set the finalize function to be executed at the termination of the program
      atexit(vftr_finalize);

      // now that initializing is done the actual hook needs
      // to be called with the appropriate arguments
      SELF_PROFILE_END_FUNCTION;
      vftr_function_entry(func, call_site);
   }
}
