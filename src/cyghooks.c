#include <stdlib.h>
#include <stdbool.h>

#include "vftrace_state.h"

#include "vftr_initialize.h"

// Define functions to redirect the function hooks, so to not make the
// function pointers globaly visible
void vftr_set_enter_func_hook(void (*function_ptr)(void*,void*)) {
   vftrace.hooks.function_hooks.enter = function_ptr;
}
void vftr_set_exit_func_hook(void (*function_ptr)(void*,void*)) {
   vftrace.hooks.function_hooks.exit = function_ptr;
}

#if defined(__x86_64__) || defined(__ve__)
void __cyg_profile_func_enter(void *func, void *call_site) {
   vftrace.hooks.function_hooks.enter(func, call_site);
}

void __cyg_profile_func_exit(void *func, void *call_site) {
   vftrace.hooks.function_hooks.exit(func, call_site);
}
#endif

#if defined(__ia64__)
// The argument func is a pointer to a pointer instead of a pointer.
void __cyg_profile_func_enter(void **func, void *call_site) {
   (void) call_site;
   vftrace.hooks.function_hooks.enter(*func, call_site);
}

void __cyg_profile_func_exit(void **func, void *call_site) {
   vftrace.hooks.function_hooks.exit(*func, call_site);
}
#endif
