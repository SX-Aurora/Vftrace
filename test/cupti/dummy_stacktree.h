#ifndef DUMMY_STACKTREE_H
#define DUMMY_STACKTREE_H

#include "stack_types.h"

void vftr_init_dummy_stacktree (uint64_t t_call);
void vftr_register_dummy_call_stack (char *stackstring, uint64_t t_call);
void vftr_register_dummy_cupti_stack (char *stackstring, int cbid, float t_ms,
                                      int mem_dir, uint64_t memcpy_bytes);
stacktree_t vftr_get_dummy_stacktree();
#endif
