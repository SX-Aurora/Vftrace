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
extern long vftr_admin_offset;
extern long vftr_samples_offset;

enum column_data_type {COL_INT, COL_DOUBLE, COL_CHAR};

typedef struct column {
	int type;
	char *header;
	int n_chars;
	int n_decimal_places;
	char format[10];
} column_t;

typedef struct format_t {
	int fid;
	int rank;
	int n_calls;
	int func_name;
	int caller_name;
	int excl_time;
	int incl_time;
	int overhead;
} format_t; 

typedef struct display_function {
    char *func_name;
    int i_orig; // The original index of the display function. Used to undo sortings by various other field values, e.g. t_avg.
    int n_calls;
    double t_avg;
    long long t_min;
    long long t_max;
    double t_sync_avg;
    long long t_sync_min;
    long long t_sync_max;
    double imbalance;
    long long this_mpi_time;
    long long this_sync_time;
    int n_stack_indices;
    int n_func_indices;
    int *stack_indices;
    int *func_indices;
    double mpi_tot_send_bytes;
    double mpi_tot_recv_bytes;
} display_function_t;

extern char *vftr_mpi_collective_function_names[];
extern int vftr_n_collective_mpi_functions;

enum sample_id {SID_ENTRY, SID_EXIT, SID_MESSAGE};

void vftr_init_vfd_file ();
void vftr_finalize_vfd_file (long long finalize_time, int signal_number);
void vftr_write_to_vfd (long long runtime, unsigned long long cycles, int stack_id, unsigned int sid);
#ifdef _MPI
double compute_mpi_imbalance (long long *all_times, double t_avg);
#endif
bool vftr_is_collective_mpi_function (char *func_name);

void vftr_store_message_info(vftr_direction dir, int count, int type_idx,
                             int type_size, int rank, int tag,
                             long long tstart, long long tend);

void vftr_get_application_times (double time0, double *total_runtime, double *sampling_overhead_time,
			  	 double *mpi_overhead_time, double *total_overhead_time, double *application_time);
void vftr_print_profile (FILE *pout, int *ntop, long long t0);
char *vftr_get_program_path ();
char *vftr_create_logfile_name (int mpi_rank, int mpi_size, char *suffix);

int vftr_filewrite_test_1 (FILE *fp_in, FILE *fp_out);
int vftr_filewrite_test_2 (FILE *fp_in, FILE *fp_out);

void vftr_print_mpi_statistics (FILE *pout);

void vftr_memory_unit(double *value, char **unit);
void vftr_time_unit (double *value, char **unit, bool for_html);

#endif
