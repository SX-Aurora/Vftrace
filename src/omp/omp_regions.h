#ifndef OMP_REGIONS_H
#define OMP_REGIONS_H

#include <stdlib.h>

void vftr_omp_region_begin(const char *name, void *addr);

void vftr_omp_region_end();

#endif
