#include <stdlib.h>
#include <stdio.h>

#include <string.h>

#include "vftrace_state.h"
#include "symbols.h"
#include "cyghooks.h"
#include "vftr_hooks.h"

void vftr_pre_hook_function_entry(void *func, void *call_site) {
   uintptr_t func_addr = (uintptr_t) func;
   char *func_name = vftr_get_name_from_address(vftrace.symboltable, func_addr);
   // skip everything in the programs execution until
   // main is called.
   // There are c++ programs that execute a lot of inconsitently
   // instrumented code before calling main.
   // This skips all this, as it can break vftrace.
   if (func_name != NULL && !strcmp(func_name, "main")) {
      // assign the appropriate function hooks to handle sampling.
      vftr_set_enter_func_hook(vftr_function_entry);
      vftr_set_exit_func_hook(vftr_function_exit);

      // execute the actual function entry hook.
      vftr_function_entry(func, call_site);
   }
}

void vftr_pre_hook_function_exit(void *func, void *call_site) {
   (void) func;
   (void) call_site;
}
