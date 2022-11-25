#include "vftrace_state.h"
#include "cyghooks.h"
#include "vftr_hooks.h"
#include "pause_hooks.h"

// pause sampling via vftrace in user code
void vftrace_pause() {
   if (vftrace.state == on) {
      vftrace.state = paused;
      // store current hooks for later restoring.
      vftrace.hooks.prepause_hooks.enter = vftrace.hooks.function_hooks.enter;
      vftrace.hooks.prepause_hooks.exit = vftrace.hooks.function_hooks.exit;
      // set the function hooks to a dummy function that does nothing
      vftr_set_enter_func_hook(vftr_function_hook_pause);
      vftr_set_exit_func_hook(vftr_function_hook_pause);
   }
}

// resume sampling via vftrace in user code
void vftrace_resume() {
   if (vftrace.state == paused) {
      vftrace.state = on;
      // set the function hooks to the previously stored hooks
      vftr_set_enter_func_hook(vftrace.hooks.prepause_hooks.enter);
      vftr_set_exit_func_hook(vftrace.hooks.prepause_hooks.exit);
      vftrace.hooks.prepause_hooks.enter = NULL;
      vftrace.hooks.prepause_hooks.exit = NULL;
   }
}
