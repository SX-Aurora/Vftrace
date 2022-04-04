#ifndef HASHING_H
#define HASHING_H

#include <stddef.h>

#include "stack_types.h"

uint64_t vftr_jenkins_murmur_64_hash(size_t length, const uint8_t* key);

void vftr_compute_stack_hashes(stacktree_t *stacktree_ptr);

#endif
