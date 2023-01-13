#ifndef VFTR_PAPI_H
#define VFTR_PAPI_H

#include <stdio.h>
#include <stdlib.h>

void vftr_show_papi_components (FILE *fp) {

void vftr_papi_init();

void vftr_papi_finalize();

long long *vftr_get_papi_counters ();

#endif
