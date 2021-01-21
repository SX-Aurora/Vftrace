#ifndef VFTR_MALLINFO_H
#define VFTR_MALLINFO_H

#define VFTR_XML_STRING_LENGTH 19

enum vftr_mallinfo_indices {MEM_FAST=11, MEM_REST=12, MEM_MMAP=13, MEM_CURRENT=14, MEM_MAX=15, MEM_TOTAL=16, MEM_PROTECTED=17};

typedef struct vftr_mallinfo {
	long fast_count;
     	long fast_size;
  	long rest_count;
 	long rest_size;
 	long mmap_count;
	long mmap_size;
	long current_count;
	long current_size;
	long max_count;
  	long max_size;
	long total_count;
	long total_size;
	long protected_count;
	long protected_size;
} vftr_mallinfo_t;

extern vftr_mallinfo_t vftr_current_mallinfo;
extern bool vftr_memtrace;

void vftr_init_mallinfo();
void vftr_get_mallinfo();

#endif

