#ifndef VFTR_VFD_UTILS_H
#define VFTR_VFD_UTILS_H

#define RECORD_LENGTH 10240

typedef struct FileHeader {
    char fileid[VFTR_FILEIDSIZE], date[24];
    long long interval;
    int threads, thread, tasks, task; 
    double runtime;
    unsigned int samplecount;
    unsigned int function_samplecount;
    unsigned int message_samplecount;
    unsigned int stackscount;
    long sampleoffset, stacksoffset;
    int n_hw_obs;
    int n_formulas;
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

extern char **vftr_hw_obs_name;
extern char **vftr_variable_name;


bool is_precise (char *s);
char *strip_trailing_asterisk (char *s);

void read_fileheader (vfd_header_t *vfd_header, FILE *fp);
void print_fileheader (FILE *fp, vfd_header_t vfd_header);
void read_stacks (FILE *fp, stack_entry_t **stacks, function_entry_t **functions,
		  unsigned int stacks_count, long stacks_offset,
		  int *n_precise_function, long *max_fp);

void read_mpi_message_sample (FILE *fp, int *direction, int *rank, int *type_index,
			      int *type_size, int *count, int *tag,
			      double *dt_start, double *dt_stop, double *rate,
                              int *callingStackID);
void skip_mpi_message_sample (FILE *fp);

void read_scenario_header (FILE *fp, int n_hw_obs, int n_formulas, bool verbose);
void skip_hw_observables (FILE *fp, int n_hw_obs);

void read_stack_sample (FILE *fp, int n_hw_obs, int *stack_id,
			long long *sample_time, long long *hw_values, long long *cycle_time);
void skip_stack_sample (FILE *fp);
void cleanup_scenario_data (int n_hw_obs);
#endif
