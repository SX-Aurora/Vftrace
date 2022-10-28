#ifndef LOGFILE_MPI_TABLE_H
#define LOGFILE_MPI_TABLE_H

#include <stdlib.h>
#include <stdio.h>

#include "collated_stack_types.h"
#include "configuration_types.h"

void vftr_write_logfile_mpi_table(FILE *fp, collated_stacktree_t stacktree,
                                  config_t config); 

#endif
