#ifndef VFTR_FILEWRITE_H
#define VFTR_FILEWRITE_H

#define VFTR_FILEIDSIZE 16
#define VFD_VERSION 2

#include <stdio.h>
#include <stdbool.h>

#include "vftr_functions.h"
#include "mpi_util_types.h"
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

enum vftr_remarks {REMARK_NONE, REMARK_OVERHEAD, REMARK_DUMMY};

typedef struct remarks {
  int id;
  struct remarks *r_next;
} remarks_t;

enum column_data_type {COL_INT, COL_DOUBLE, COL_CHAR_LEFT, COL_CHAR_RIGHT, COL_MEM, COL_TIME};
enum separator_t {SEP_NONE, SEP_MID, SEP_LAST};

typedef struct column {
	int col_type;
	char *header;
   	char *group_header;
	int n_chars;
	int n_chars_opt_1;
	int n_chars_opt_2;
	int n_decimal_places;
	int separator_type;
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
    int stack_id;
    int func_id;
    int n_calls;
    double t_avg;
    long long t_min;
    int rank_min;
    long long t_max;
    int rank_max;
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
    bool is_mpi;
    double mpi_tot_send_bytes;
    double mpi_tot_recv_bytes;
    bool properly_terminated;
} display_function_t;

enum log_time_type {TOTAL_TIME, SAMPLING_OVERHEAD, MPI_OVERHEAD, TOTAL_OVERHEAD, APP_TIME};
typedef struct vftr_prof_times {
   long long t_usec[5];
   long long t_sec[5];
} vftr_prof_times_t;

display_function_t **vftr_create_display_functions (bool display_sync_time, int *n_display_funcs, bool use_all);

enum sample_id {SID_ENTRY, SID_EXIT, SID_MESSAGE};

void vftr_init_vfd_file ();
void vftr_finalize_vfd_file (long long finalize_time);
void vftr_write_to_vfd (long long runtime, profdata_t *prof_current, int stack_id, unsigned int sid);
#ifdef _MPI
double vftr_compute_mpi_imbalance (long long *all_times, double t_avg);
#endif

#ifdef _MPI
// store some message information for use in the log file
void vftr_log_message_info(vftr_direction dir, int count, int type_idx,
                           int type_size, int rank, int tag,
                           long long tstart, long long tend);
#endif

void vftr_store_message_info(vftr_direction dir, int count, int type_idx,
                             int type_size, int rank, int tag,
                             long long tstart, long long tend,
                             int callingStackID);

vftr_prof_times_t vftr_get_application_times_usec (long long time0, bool include_t[5]);
int vftr_count_func_indices_up_to_truncate (function_t **func_table, long long runtime_usec);

void vftr_print_profile (FILE *fp_log, function_t **sorted_func_table, int n_func_indices,
                         vftr_prof_times_t prof_times,
                         int n_display_functions, display_function_t **display_functions);

void vftr_print_html_profile (FILE *f_html, function_t **sorted_func_table,
                              const int n_func_indices, vftr_prof_times_t prof_times, 
                              const int n_display_functions, display_function_t **display_functions,
                              long long time0);

char *vftr_get_program_path ();
char *vftr_create_logfile_name (int mpi_rank, int mpi_size, char *suffix);

void vftr_print_function_statistics (FILE *fp_log, display_function_t **display_functions, int n_display_functions, bool print_this_rank);

void vftr_print_function_statistics_html (display_function_t **display_functions, vftr_prof_times_t prof_times, int n_display_functions, bool print_this_rank);
void vftr_memory_unit(double *value, char **unit);
char *vftr_memory_unit_string (double value, int n_decimal_places);
void vftr_time_unit (double *value, char **unit, bool for_html);

void vftr_prof_column_init (const char *name, char *group_header, int n_decimal_places, int col_type, int sep_type, column_t *c);
void vftr_prof_column_set_n_chars (void *value, void *opt_1, void *opt_2, column_t *c, int *stat);
void vftr_prof_column_print (FILE *fp, column_t c, void *value, void *opt_1, void *opt_2);
int vftr_get_tablewidth_from_columns (column_t *columns, int n_columns, bool use_separators);

#endif
