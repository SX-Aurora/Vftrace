#ifndef HOOK_TYPES_H
#define HOOK_TYPES_H

// function pointers which will be called by the cyg-hooks
// On the first function entry vftrace will be initialized.
// After that the function pointers will be redirected to the
// actual hook functionality or a dummy function if vftrace is off.
typedef struct {
   void (*enter)(void*, void*);
   void (*exit)(void*, void*);
} function_hook_t;

typedef struct {
   function_hook_t function_hooks;
   function_hook_t prepause_hooks;
} hooks_t;

#endif
