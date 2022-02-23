#ifdef _DEBUG
#include <stdio.h>
#endif

#include "off_hooks.h"
#include "cyghooks.h"
#include "vftr_hooks.h"
#include "vftrace_state.h"
#include "environment.h"

void vftr_initialize(void *func, void *caller) {
   // First step is to parse the relevant environment variables
   vftrace.environment = vftr_read_environment();
   vftr_environment_free(&(vftrace.environment));
   
   // update the vftrace state
   vftrace.state = initialized;








   // redirect the function entry and exit hooks to call the appropriate functions
   if (vftrace.state == off) {
      // use a dummy function that does nothing
      vftr_set_enter_func_hook(vftr_function_hook_off);
      vftr_set_exit_func_hook(vftr_function_hook_off);
   } else {
      // assign the appropriate function hooks.
      vftr_set_enter_func_hook(vftr_function_entry);
      vftr_set_exit_func_hook(vftr_function_exit);

      // now that initializing is done the actual hook needs
      // to be called with the appropriate arguments
      vftr_function_entry(func, caller);
   }
}
