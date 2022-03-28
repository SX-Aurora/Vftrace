#ifndef SORTING_H
#define SORTING_H

#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#include "symbols.h"

void vftr_sort_integer(int *list, int n, bool ascending);

void vftr_sort_double(double *list, int n, bool ascending);

void vftr_sort_double_copy (double *d_array, int n, bool ascending, double *d_copy);

// sorts a list of unsigned 64 bit integer with linear scaling radix sort
// one bit at a time
void vftr_radixsort_uint64(int n, uint64_t *list);

// sort the symboltable with a linear scaling radixsort
void vftr_radixsort_symboltable(unsigned int nsymb, symbol_t *symbols);

#endif
