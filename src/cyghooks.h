#ifndef CYGHOOKS_H
#define CYGHOOKS_H

// Define functions to redirect the function hooks, so to not make the
// function pointers globaly visible
void vftr_set_enter_func_hook(void (*function_ptr)(void*,void*));
void vftr_set_exit_func_hook(void (*function_ptr)(void*,void*));

#endif
