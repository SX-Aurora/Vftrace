#ifndef VFTR_VFD_UTILS_H
#define VFTR_VFD_UTILS_H

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

void read_fileheader (vfd_header_t *vfd_header, FILE *fp);


#endif
