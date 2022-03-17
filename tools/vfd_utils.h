#ifndef VFD_UTILS_H
#define VFD_UTILS_H

#include "vfd_types.h"

vfd_header_t read_vfd_header(FILE *fp);

void free_vfd_header(vfd_header_t *vfd_header);

void print_vfd_header(FILE *fp, vfd_header_t vfd_header);

stack_t *read_stacklist(FILE *fp, long int stacks_offset,
                        unsigned int nstacks);

void free_stacklist(unsigned int nstacks, stack_t *stacklist);

#endif
