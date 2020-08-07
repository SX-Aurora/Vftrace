#ifndef VFTR_FILEWRITE_H
#define VFTR_FILEWRITE_H

#define VFTR_FILEIDSIZE 16
#define VFD_VERSION 1

#include <stdio.h>
#include "vftr_mpi_utils.h"

// File pointer of the log file
extern FILE *vftr_log;

// Individual vftrace-internal file id
extern char vftr_fileid[VFTR_FILEIDSIZE];

// The next time step where a snapshot is written to the vfd file
extern long long vftr_nextsampletime;

extern char *vftr_program_path;

// The basename of Vftrace log files
extern char *vftr_logfile_name;

extern FILE *vftr_vfd_file;

// TODO: Explain
extern unsigned int vftr_admin_offset;
extern unsigned int vftr_samples_offset;

typedef struct format_t {
	int fid;
	int rank;
	int n_calls;
	int func_name;
	int caller_name;
	int excl_time;
	int incl_time;
} format_t; 

enum sample_id {SID_ENTRY, SID_EXIT, SID_MESSAGE};

void vftr_init_vfd_file ();
void vftr_finalize_vfd_file (long long finalize_time, int signal_number);
void vftr_write_to_vfd (long long runtime, unsigned long long cycles, int stack_id, unsigned int sid);
void vftr_store_message_info(vftr_direction dir, int count, int type_idx,
                             int type_size, int rank, int tag,
                             long long tstart, long long tend);

void vftr_print_profile (FILE *pout, int *ntop, long long t0);
char *vftr_get_program_path ();
char *vftr_create_logfile_name (int mpi_rank, int mpi_size, char *suffix);

int vftr_filewrite_test_1 (FILE *fp);

#endif
