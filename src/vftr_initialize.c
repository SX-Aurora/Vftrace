#include <stdlib.h>

#ifdef _OMP
#include "start_tool.h"
#endif

#ifdef _CUDA
#include "cupti_init_final.h"
#include "cupti_vftr_callbacks.h"
#endif

#ifdef _ACCPROF
#include "accprof_init_final.h"
#include "accprof_callbacks.h"
#endif

#include "hwprof_init_final.h"
#include "self_profile.h"
#include "timer.h"
#include "off_hooks.h"
#include "cyghooks.h"
#include "pre_hooks.h"
#include "vftr_hooks.h"
#include "vftrace_state.h"
#include "configuration.h"
#include "configuration_assert.h"
#include "symbols.h"
#include "vftr_finalize.h"
#include "processes.h"
#include "sampling.h"
#include "signal_handling.h"

void vftr_initialize(void *func, void *call_site) {
   INIT_SELF_PROF_VFTRACE;
   SELF_PROFILE_START_FUNCTION;
   // First step is to initialize the reference timer
   vftr_set_local_ref_time();

   vftr_setup_signals();

   // parse the config file
   vftrace.config = vftr_read_config();

   if (vftrace.config.off.value) {
      // update the vftrace state
      vftrace.state = off;
      // free the config to avoid memory leaks
      vftr_config_free(&(vftrace.config));
      // set the function hooks to a dummy function that does nothing
      vftr_set_enter_func_hook(vftr_function_hook_off);
      vftr_set_exit_func_hook(vftr_function_hook_off);
#ifdef _ACCPROF
      // Do not register any OpenACC callbacks and deactivate potentially registered ones.
      vftr_veto_accprof_callbacks ();
#endif
      SELF_PROFILE_END_FUNCTION;
      FINALIZE_SELF_PROF_VFTRACE;
   } else {
      // update the vftrace state
      vftrace.state = on;

      // set start time string
      vftrace.timestrings.start_time = vftr_get_date_str();

      // check configuration consistency
      vftr_config_assert(stderr, vftrace.config);

      // read the symbol table of the executable and its libraries
      vftrace.symboltable = vftr_read_symbols();
      vftr_symboltable_determine_preciseness(&(vftrace.symboltable),
                                             vftrace.config.sampling.precise_functions.regex);
      vftr_symboltable_strip_fortran_module_name(&(vftrace.symboltable),
                                                 vftrace.config.strip_module_names.value);
#ifdef _LIBERTY
      vftr_symboltable_demangle_cxx_name(&(vftrace.symboltable),
                                         vftrace.config.demangle_cxx.value);
#endif

      // We need to init PAPI before the first profile is allocated, because
      // it needs the number of registered PAPI counters.
      vftr_hwprof_init (vftrace.config);

      // initialize the dynamic process data
      vftrace.process = vftr_new_process();

      // initialize possible sampling
      vftrace.sampling = vftr_new_sampling(vftrace.config);

      if (vftrace.config.include_cxx_prelude.value) {
         // assign the vftr_function hooks to the function entry and exit hooks
         // to start profiling right away
         vftr_set_enter_func_hook(vftr_function_entry);
         vftr_set_exit_func_hook(vftr_function_exit);
      } else {
         // assign pre_hooks to the function entry and exit hooks
         // There are c++ programs that execute a lot of inconsistently
         // instrumented code before calling main.
         // That is skipped by the pre_hook assignment.
         vftr_set_enter_func_hook(vftr_pre_hook_function_entry);
         vftr_set_exit_func_hook(vftr_pre_hook_function_exit);
      }

      // trick the linker into including extra symbols
#ifdef _OMP
      // omp callback symbols
      (void) ompt_start_tool(0, NULL);
#endif

#ifdef _CUDA
      (void)vftr_init_cupti(vftr_cupti_event_callback);
#endif

#ifdef _ACCPROF
      if (!vftrace.config.accprof.active.value) {
         vftr_veto_accprof_callbacks(); 
      } else {
         vftr_init_accprof();
      }
#endif

      // set the finalize function to be executed at the termination of the program
      atexit(vftr_finalize);

      if (vftrace.config.include_cxx_prelude.value) {
         // execute the actual function entry hook.
         SELF_PROFILE_END_FUNCTION;
         vftr_function_entry(func, call_site);
      } else {
         // now that initializing is done the actual hook needs
         // to be called with the appropriate arguments
         SELF_PROFILE_END_FUNCTION;
         vftr_pre_hook_function_entry(func, call_site);
      }
   }
}
