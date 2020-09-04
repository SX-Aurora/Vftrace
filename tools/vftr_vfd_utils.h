#ifndef VFTR_VFD_UTILS_H
#define VFTR_VFD_UTILS_H

#define RECORD_LENGTH 10240

typedef struct FileHeader {
    char fileid[VFTR_FILEIDSIZE], date[24];
    long long interval;
    int threads, thread, tasks, task; 
    union { double d; unsigned long long l; } cycletime, runtime;
    long long inittime;
    unsigned int samplecount, sampleoffset;
    unsigned int stackscount, stacksoffset;
    unsigned int reserved;
    int n_hw_obs;
} vfd_header_t;

typedef struct FunctionEntry {
    char  *name;
    double elapse_time;
} function_entry_t;

typedef struct StackEntry {
    char *name;
    int caller;
    int levels;
    double entry_time;
    int fun;
    bool precise;
} stack_entry_t;

bool is_precise (char *s);
char *strip_trailing_asterisk (char *s);

void read_fileheader (vfd_header_t *vfd_header, FILE *fp);
void read_stacks (FILE *fp, stack_entry_t **satcks, function_entry_t **functions,
		  unsigned int stacks_count, unsigned int stacks_offset,
		  int *n_precise_function, long *max_fp);

void read_mpi_message_sample (FILE *fp, int *direction, int *rank, int *type_index,
			      int *type_size, int *count, int *tag,
			      double *dt_start, double *dt_stop, double *rate);
void skip_mpi_message_sample (FILE *fp);

void init_hw_observables (FILE *fp, int n_hw_obs, double **hw_values);
void skip_hw_observables (FILE *fp, int n_hw_obs);

void read_stack_sample (FILE *fp, int n_hw_obs, int *stack_id,
			long long *sample_time, double **hw_values);
void skip_stack_sample (FILE *fp);
#endif
