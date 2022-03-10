#ifndef PROCESSES_H
#define PROCESSES_H

#include "process_types.h"

process_t vftr_new_process();

void vftr_process_free(process_t *process_ptr);

#endif
