#ifndef SORTING_H
#define SORTING_H

#include <stdint.h>
#include <stdlib.h>

#include "symbols.h"

// sorts a list of unsigned 64 bit integer with linear scaling radix sort
// one bit at a time
void vftr_radixsort_uint64(int n, uint64_t *list);

// sort the symboltable with a linear scaling radixsort
void vftr_radixsort_symboltable(unsigned int nsymb, symbol_t *symbols);

#endif
