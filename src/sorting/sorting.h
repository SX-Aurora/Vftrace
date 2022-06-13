#ifndef SORTING_H
#define SORTING_H

#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#include "symbols.h"

// sorts a list of integers with linear scaling radix sort
// one bit at a time
void vftr_radixsort_int8(int n, int8_t *list);
void vftr_radixsort_int16(int n, int16_t *list);
void vftr_radixsort_int32(int n, int32_t *list);
void vftr_radixsort_int64(int n, int64_t *list);

void vftr_radixsort_char(int n, char *list);
void vftr_radixsort_short(int n, short *list);
void vftr_radixsort_int(int n, int *list);
void vftr_radixsort_long(int n, long *list);
void vftr_radixsort_longlong(int n, long long *list);

// sorts a list of unsigned 64 bit integer with linear scaling radix sort
// one bit at a time
void vftr_radixsort_uint8(int n, uint8_t *list);
void vftr_radixsort_uint16(int n, uint16_t *list);
void vftr_radixsort_uint32(int n, uint32_t *list);
void vftr_radixsort_uint64(int n, uint64_t *list);

void vftr_radixsort_unsignedchar(int n, unsigned char *list);
void vftr_radixsort_unsignedshort(int n, unsigned short *list);
void vftr_radixsort_unsignedint(int n, unsigned int *list);
void vftr_radixsort_unsignedlong(int n, unsigned long *list);
void vftr_radixsort_unsignedlonglong(int n, unsigned long long *list);

// sorts a list of doubles with linear scaling radix sort
// one bit at a time
void vftr_radixsort_float(int n, float *list);
void vftr_radixsort_double(int n, double *list);

// sort the symboltable with a linear scaling radixsort
void vftr_radixsort_symboltable(unsigned int nsymb, symbol_t *symbols);

#endif
