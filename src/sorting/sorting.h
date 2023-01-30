#ifndef SORTING_H
#define SORTING_H

#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#include "symbols.h"
#include "stack_types.h"
#include "collated_stack_types.h"
#include "configuration_types.h"

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
void vftr_apply_perm_longlong(int n, long long *list, int *perm);
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

// sorts a list of floats or doubles with linear scaling radix sort
// one bit at a time
// Record the sorting process in a permutation for later use
void vftr_sort_perm_float(int n, float *list, int **perm, bool ascending);
void vftr_apply_perm_float(int n, float *list, int *perm);
void vftr_sort_perm_double(int n, double *list, int **perm, bool ascending);
void vftr_apply_perm_double(int n, double *list, int *perm);

void vftr_apply_perm_charptr (int n, char **list, int *perm);

// sort the symboltable with a linear scaling sort
void vftr_sort_symboltable(unsigned int nsymb, symbol_t *symbols);

// sort the stacks based on a set configuration variable
void vftr_apply_perm_stackptr(int n, vftr_stack_t **list, int *perm);
vftr_stack_t **vftr_sort_stacks_for_prof(config_t config,
                                    stacktree_t stacktree);
#ifdef _MPI
void vftr_sort_collated_stacks_for_mpiprof(config_t config,
                                           int nselected_stacks,
                                           collated_stack_t **selected_stacks);
void vftr_sort_stacks_for_mpiprof(config_t config,
                                  int nselected_stacks,
                                  vftr_stack_t **selected_stacks);
#endif

#ifdef _CUDA
vftr_stack_t **vftr_sort_stacks_for_cuda (config_t config, stacktree_t stacktree);
collated_stack_t **vftr_sort_collated_stacks_for_cuda (config_t config, collated_stacktree_t stacktree);
#endif

#ifdef _ACCPROF
vftr_stack_t **vftr_sort_stacks_for_accprof (config_t config, stacktree_t stacktree);
collated_stack_t **vftr_sort_collated_stacks_for_accprof (config_t config, collated_stacktree_t stacktree);
#endif

vftr_stack_t **vftr_sort_stacks_hwprof_obs (config_t config, stacktree_t stacktree);
collated_stack_t **vftr_sort_collated_stacks_hwprof_obs (config_t config, collated_stacktree_t stacktree);

// sort the collated stacks based on a set configuration variable
void vftr_apply_perm_collated_stackptr(int n, collated_stack_t **list, int *perm);
collated_stack_t **vftr_sort_collated_stacks_for_prof(config_t config,
                                                      collated_stacktree_t stacktree);
collated_stack_t **vftr_sort_collated_stacks_tmax (config_t config, collated_stacktree_t stacktree);

#endif
