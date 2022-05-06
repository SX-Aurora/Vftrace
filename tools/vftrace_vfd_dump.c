#include <stdlib.h>
#include <stdio.h>

#include <assert.h>
#include <string.h>

#include "vfd_dump_cmd_options.h"
#include "vfd_types.h"
#include "vfd_utils.h"

int main(int argc, char **argv) {

   // parse command line options
   cmd_options_t options = parse_command_line_options(argc, argv);

   // output filepath
   FILE *out_fp = NULL;
   bool outfp_is_file = false;
   if (options.output_filename == NULL) {
      out_fp = stdout;
      outfp_is_file = false;
   } else if (strcmp(options.output_filename, "stderr") == 0) {
      out_fp = stderr;
      outfp_is_file = false;
   } else if (strcmp(options.output_filename, "stdout") == 0) {
      out_fp = stdout;
      outfp_is_file = false;
   } else {
      out_fp = fopen(options.output_filename, "w");
      outfp_is_file = true;
   }

   // vfd filepath
   FILE *vfd_fp = NULL;
   if (options.vfd_filename == NULL) {
      fprintf(stderr, "No vfd-file specified. check \"--help\" for help.\n");
      abort();
   } else {
      vfd_fp = fopen(options.vfd_filename, "r");
      assert(vfd_fp);
   }

   vfd_header_t vfd_header = read_vfd_header(vfd_fp);
   print_vfd_header(out_fp, vfd_header);

   if (!options.only_header) {
      stack_t *stacklist = read_stacklist(vfd_fp,
                                          vfd_header.stacks_offset,
                                          vfd_header.nstacks);

      fprintf(out_fp, "\n");
      print_stacklist(out_fp, vfd_header.nstacks, stacklist);

      if (!options.skip_samples) {
         fprintf(out_fp, "\n");
         print_samples(vfd_fp, out_fp, vfd_header, stacklist);
      }

      free_stacklist(vfd_header.nstacks, stacklist);
   }

   free_vfd_header(&vfd_header);

   if (outfp_is_file) {
      fclose(out_fp);
   }
   fclose(vfd_fp);
}
