#ifndef VFTR_MALLINFO_H
#define VFTR_MALLINFO_H

enum vftr_mallinfo_method {MEM_MALLOC_INFO, MEM_VMRSS};

typedef struct vftr_mallinfo {
	long fast_count;
     	long fast_size;
  	long rest_count;
 	long rest_size;
 	long mmap_count;
	long long mmap_size;
	long current_count;
	long current_size;
	long max_count;
  	long max_size;
	long total_count;
	long total_size;
	long protected_count;
	long protected_size;
} vftr_mallinfo_t;

extern int vftr_xml_string_length;
extern int vftr_meminfo_method;
extern bool vftr_memtrace;
extern int vftr_mmap_xml_index;
extern long long vftr_mallinfo_ovhd;
extern long long vftr_mallinfo_post_ovhd;
extern FILE *vftr_fp_selfstat;

void vftr_init_mallinfo();
void vftr_finalize_mallinfo();
void vftr_get_memtrace(bool verbose);

long long vftr_get_vmrss (bool verbose);

#endif

