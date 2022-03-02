#include <stdlib.h>
#ifdef _DEBUG
#include <stdio.h>
#include <time.h>
#endif

#include "off_hooks.h"
#include "cyghooks.h"
#include "vftr_hooks.h"
#include "vftrace_state.h"
#include "environment.h"
#include "symbols.h"
#include "finalize.h"

void vftr_initialize(void *func, void *call_site) {
   // First step is to initialize the reference timer
   reftime_t reftime = vftr_set_local_ref_time();
#ifdef _DEBUG
   fprintf(stderr, "Vftrace initilized at ");
   vftr_print_date_str(stderr);
   fprintf(stderr, "\n");
#endif
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
      vftrace.state = initialized;

      // read the symbol table of the executable and its libraries
      vftrace.symboltable = vftr_read_symbols();

      // assign the appropriate function hooks to handle sampling.
      vftr_set_enter_func_hook(vftr_function_entry);
      vftr_set_exit_func_hook(vftr_function_exit);

      // set the finalize function to be executed at the termination of the program
      atexit(vftr_finalize);

      // now that initializing is done the actual hook needs
      // to be called with the appropriate arguments
      vftr_function_entry(func, call_site);
   }
}
