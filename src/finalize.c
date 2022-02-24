#ifdef _DEBUG
#include <stdio.h>
#endif

#include "off_hooks.h"
#include "cyghooks.h"
#include "vftr_hooks.h"
#include "vftrace_state.h"

void vftr_finalize() {
   // update the vftrace state
   vftrace.state = finalized;








   // redirect the function entry and exit hooks to deactivate them
   // use a dummy function that does nothing
   vftr_set_enter_func_hook(vftr_function_hook_off);
   vftr_set_exit_func_hook(vftr_function_hook_off);
}
