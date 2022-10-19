#ifndef VFTRACE_H
#define VFTRACE_H

#ifdef __cplusplus
extern "C" {
#endif
// Mark start of instrumented region
// name ist the Region name to be used in the profile
void vftrace_region_begin(const char *name);

// Mark end of instrumented region
// name ist the Region name to be used in the profile
void vftrace_region_end(const char *name);

//void vftrace_allocate (const char *name, int n);
//
// obtain the stack string as char pointer (needs deallocation after use)
char *vftrace_get_stack();

// pause and resume sampling via vftrace in user code
void vftrace_pause();
void vftrace_resume();

//void vftrace_show_callstack();
//int vftrace_get_stacktree_size();

#ifdef __cplusplus
}
#endif
#endif
