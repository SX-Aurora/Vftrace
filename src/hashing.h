#ifndef HASHING_H
#define HASHING_H

#include "stack_types.h"

uint64_t vftr_jenkins_murmur_64_hash(size_t length, const uint8_t* key);

void vftr_compute_stack_hashes(int nstacks, stack_t *stacks);

#endif
