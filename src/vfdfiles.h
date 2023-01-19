#ifndef VFDFILES_H
#define VFDFILES_H

#include "configuration_types.h"
#include "process_types.h"
#include "hwprof_state_types.h"
#include "timer_types.h"

#define VFD_VERSION 3

// At the initialization of vftrace the mpi-rank and comm-size is
// not known for paralle programs.
// Thus a preliminary vfdfile is created:
// <basename>_<pid>.tmpvfd
// In the finalization it will be moved to its proper name
// <basename>_<mpi-rank>.vfd
char *vftr_get_preliminary_vfdfile_name(config_t config);

char *vftr_get_vfdfile_name(config_t config, int rankID, int nranks);

FILE *vftr_open_vfdfile(char *filename);

char *vftr_attach_iobuffer_vfdfile(FILE *fp, config_t config,
                                   size_t *buffersize);

int vftr_rename_vfdfile(char *prelim_name, char *final_name);

void vftr_write_incomplete_vfd_header(sampling_t *sampling);

void vftr_update_vfd_header(sampling_t *sampling,
                            process_t process,
                            time_strings_t timestrings,
                            double runtime);
void vftr_write_vfd_stacks(sampling_t *sampling, stacktree_t stacktree);

void vftr_write_vfd_threadtree(sampling_t *sampling, threadtree_t threadtree);

void vftr_write_vfd_hwprof_info(sampling_t *sampling, hwprof_state_t state);

void vftr_write_vfd_function_sample(sampling_t *sampling, sample_kind kind,
                                    int stackID, long long timestamp,
                                    long long *hwcounters);

#endif
