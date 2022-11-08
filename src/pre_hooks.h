#ifndef PRE_HOOKS_H
#define PRE_HOOKS_H

#include <stdlib.h>

void vftr_pre_hook_function_entry(void *func, void *call_site);

void vftr_pre_hook_function_exit(void *func, void *call_site);

#endif
