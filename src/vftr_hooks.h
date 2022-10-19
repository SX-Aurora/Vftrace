#ifndef VFTR_HOOKS_H
#define VFTR_HOOKS_H

#include <stdbool.h>

void vftr_function_entry(void *func, void *call_site);

void vftr_function_exit(void *func, void *call_site);

#endif
