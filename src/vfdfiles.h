#ifndef VFDFILES_H
#define VFDFILES_H

#include "environment_types.h"

#define VFD_VERSION 2

// At the initialization of vftrace the mpi-rank and comm-size is
// not known for paralle programs.
// Thus a preliminary vfdfile is created:
// <basename>_<pid>.tmpvfd
// In the finalization it will be moved to its proper name
// <basename>_<mpi-rank>.vfd
char *vftr_get_preliminary_vfdfile_name(environment_t environment);

char *vftr_get_vfdfile_name(environment_t environment, int rankID, int nranks);

FILE *vftr_open_vfdfile(char *filename);

char *vftr_attach_iobuffer_vfdfile(FILE *fp, environment_t environment);

int vftr_rename_vfdfile(char *prelim_name, char *final_name);

#endif
