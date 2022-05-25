#ifndef VFD_DUMP_OPT_TYPES_H
#define VFD_DUMP_OPT_TYPES_H

#include <stdbool.h>

#include <argp.h>

typedef struct {
   char *output_filename;
   char *vfd_filename;
   bool only_header;
   bool skip_samples;
   bool show_threadtree;
} cmd_options_t;

enum cmd_opt_IDs {
   output_filename_ID = 1024,
   only_header_ID,
   skip_samples_ID,
   show_threadtree_ID
};

#endif
