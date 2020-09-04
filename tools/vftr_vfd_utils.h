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
    int n_perf_types;
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

void read_fileheader (vfd_header_t *vfd_header, FILE *fp);
void read_stacks (FILE *fp, stack_entry_t **satcks, function_entry_t **functions,
		  unsigned int stacks_count, unsigned int stacks_offset,
		  int *n_precise_function, long *max_fp,
		  bool remove_asterisks);


#endif
