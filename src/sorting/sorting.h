#ifndef SORTING_H
#define SORTING_H

#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#include "symbols.h"
#include "stack_types.h"
#include "collated_stack_types.h"
#include "environment_types.h"

// sorts a list of integers with linear scaling radix sort
// one bit at a time
void vftr_sort_int8(int n, int8_t *list, bool ascending);
void vftr_sort_int(int n, int *list, bool ascending);
void vftr_sort_longlong(int n, long long *list, bool ascending);

// sorts a list of integers with linear scaling radix sort
// one bit at a time
// Record the sorting process in a permutation for later use
void vftr_sort_perm_int8(int n, int8_t *list, int **perm, bool ascending);
void vftr_apply_perm_int8(int n, int8_t *list, int *perm);
void vftr_sort_perm_longlong(int n, long long *list, int **perm, bool ascending);
void vftr_sort_perm_int(int n, int *list, int **perm, bool ascending);

// sorts a list of unsigned 64 bit integer with linear scaling radix sort
// one bit at a time
void vftr_sort_uint8(int n, uint8_t *list, bool ascending);
void vftr_sort_uint64(int n, uint64_t *list, bool ascending);

// sorts a list of integers with linear scaling radix sort
// one bit at a time
// Record the sorting process in a permutation for later use
void vftr_sort_perm_uint8(int n, uint8_t *list, int **perm, bool ascending);
void vftr_apply_perm_uint8(int n, uint8_t *list, int *perm);

// sorts a list of floats with linear scaling radix sort
// one bit at a time
void vftr_sort_float(int n, float *list, bool ascending);
void vftr_sort_double(int n, double *list, bool ascending);

// sorts a list of floats with linear scaling radix sort
// one bit at a time
// Record the sorting process in a permutation for later use
void vftr_sort_perm_float(int n, float *list, int **perm, bool ascending);
void vftr_apply_perm_float(int n, float *list, int *perm);

// sort the symboltable with a linear scaling sort
void vftr_sort_symboltable(unsigned int nsymb, symbol_t *symbols);

// sort the stacks based on a set environment variable
void vftr_apply_perm_stackptr(int n, stack_t **list, int *perm);
stack_t **vftr_sort_stacks_for_prof(environment_t environment,
                                    stacktree_t stacktree);
#ifdef _MPI
void vftr_sort_collated_stacks_for_mpiprof(environment_t environment,
                                           int nselected_stacks,
                                           collated_stack_t **selected_stacks);
void vftr_sort_stacks_for_mpiprof(environment_t environment,
                                  int nselected_stacks,
                                  stack_t **selected_stacks);
#endif

#ifdef _CUPTI
stack_t **vftr_sort_stacks_for_cupti (environment_t environment, stacktree_t stacktree);
collated_stack_t **vftr_sort_collated_stacks_for_cupti (environment_t environment, collated_stacktree_t stacktree);
#endif

// sort the collated stacks based on a set environment variable
void vftr_apply_perm_collated_stackptr(int n, collated_stack_t **list, int *perm);
collated_stack_t **vftr_sort_collated_stacks_for_prof(environment_t environment,
                                                      collated_stacktree_t stacktree);

#endif
