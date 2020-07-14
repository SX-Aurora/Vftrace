#ifndef VFTR_LOADBALANCE_H
#define VFTR_LOADBALANCE_H

#include "vftr_functions.h"
#include "vftr_timer.h"

callsTime_t **vftr_get_loadbalance_info (function_t **ftable);
void vftr_print_loadbalance (callsTime_t **gct, int rankMin,
			     int nextRankMin, FILE *pout, int *list, int *nlist);

#endif
