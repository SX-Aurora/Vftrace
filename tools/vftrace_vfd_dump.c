#include <stdlib.h>
#include <stdio.h>

#include <assert.h>
#include <string.h>

#include "signal_handling.h"

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
      vftr_abort(0);
   } else {
      vfd_fp = fopen(options.vfd_filename, "r");
      assert(vfd_fp);
   }

   vfd_header_t vfd_header = read_vfd_header(vfd_fp);
   print_vfd_header(out_fp, vfd_header);

   if (!options.only_header) {
      if (options.show_threadtree) {
         thread_t *threadtree = read_threadtree(vfd_fp,
                                                vfd_header.threadtree_offset,
                                                vfd_header.nthreads);
         fprintf(out_fp, "\n");
         print_threadtree(out_fp, threadtree);
         free_threadtree(vfd_header.nthreads, threadtree);
      }

      vftr_stack_t *stacklist = read_stacklist(vfd_fp,
                                          vfd_header.stacks_offset,
                                          vfd_header.nstacks);
      fprintf(out_fp, "\n");
      print_stacklist(out_fp, vfd_header.nstacks, stacklist);

      if (!options.skip_samples) {
         fprintf(out_fp, "\n");
         print_samples(vfd_fp, out_fp, vfd_header, stacklist);
      }

      free_stacklist(vfd_header.nstacks, stacklist);

      char **hwc_names = (char**)malloc(vfd_header.n_hw_counters * sizeof(char*));
      char **symbols = (char**)malloc(vfd_header.n_hw_counters * sizeof(char*));
      char **obs_names = (char**)malloc(vfd_header.n_hw_observables * sizeof(char*));
      char **formulas = (char**)malloc(vfd_header.n_hw_observables * sizeof(char*));
      char **units = (char**)malloc(vfd_header.n_hw_observables * sizeof(char*));
      read_hwprof (vfd_fp, vfd_header.hwprof_offset,
                       vfd_header.n_hw_counters, vfd_header.n_hw_observables,
                       hwc_names, symbols,
                       obs_names, formulas, units);
      printf ("Counter names & symbols: \n");
      for (int i = 0; i < vfd_header.n_hw_counters; i++) {
         printf ("%s -> %s\n", hwc_names[i], symbols[i]);
      }

      printf ("Observables: \n");
      for (int i = 0; i < vfd_header.n_hw_observables; i++) {
         printf ("%s: %s [%s]\n", obs_names[i], formulas[i],
                 units[i] != NULL ? units[i] : "");
      }
   }

   free_vfd_header(&vfd_header);

   if (outfp_is_file) {
      fclose(out_fp);
   }
   fclose(vfd_fp);
}
