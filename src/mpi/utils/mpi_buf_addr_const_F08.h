#include <stdbool.h>

#ifndef VFTR_MPI_BUF_ADDR_CONST_F_H
#define VFTR_MPI_BUF_ADDR_CONST_F_H

bool vftr_is_F08_MPI_BOTTOM(const void *addr);

bool vftr_is_F08_MPI_IN_PLACE(const void *addr);

#endif
