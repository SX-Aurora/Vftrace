#ifndef VFD_UTILS_H
#define VFD_UTILS_H

#include <stdlib.h>
#include "vfd_types.h"

vfd_header_t read_vfd_header(FILE *fp);

void free_vfd_header(vfd_header_t *vfd_header);

void print_vfd_header(FILE *fp, vfd_header_t vfd_header);

stack_t *read_stacklist(FILE *fp, long int stacks_offset,
                        unsigned int nstacks);

void free_stacklist(unsigned int nstacks, stack_t *stacklist);

void print_stacklist(FILE *fp, unsigned int nstacks, stack_t *stacklist);

thread_t *read_threadtree(FILE *vfd_fp, long int threadtree_offset,
                          int nthreads);

void free_threadtree(int nthreads, thread_t *threadtree);

void print_threadtree(FILE *out_fp, thread_t *threadtree);

void print_samples(FILE *vfd_fp, FILE *fp_out,
                   vfd_header_t vfd_header, stack_t *stacklist);


#endif
