#include "vftrace_state.h"
#include "cyghooks.h"
#include "vftr_hooks.h"
#include "pause_hooks.h"

// pause sampling via vftrace in user code
void vftrace_pause() {
   if (vftrace.state == on) {
      vftrace.state = paused;
      // set the function hooks to a dummy function that does nothing
      vftr_set_enter_func_hook(vftr_function_hook_pause);
      vftr_set_exit_func_hook(vftr_function_hook_pause);
   }
}

// resume sampling via vftrace in user code
void vftrace_resume() {
   if (vftrace.state == paused) {
      vftrace.state = on;
      // set the function hooks to a dummy function that does nothing
      vftr_set_enter_func_hook(vftr_function_entry);
      vftr_set_exit_func_hook(vftr_function_exit);
   }
}
