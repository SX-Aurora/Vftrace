#include <stdlib.h>
#include <stdio.h>

#include <string.h>
#include <argp.h>

#include <config.h>

#include "vfd_dump_opt_types.h"

const char *argp_program_version = PACKAGE_VERSION;

const char *argp_program_bug_address = PACKAGE_BUGREPORT;

// program documentation
static char doc[] = "Dump contents of vfd files from vftrace in human readable format.";

static struct argp_option possible_options[] = {
   {
      "output",
      output_filename_ID,
      "stdout/stderr/<filename>",
      0,
      "output stream, or filename for output",
      0
   }, {
      "only_header",
      only_header_ID,
      0,
      0,
      "print only header information",
      0
   }, {
      "skip_samples",
      skip_samples_ID,
      0,
      0,
      "do not print the samples",
      0
   }, {
      "show_threadtree",
      show_threadtree_ID,
      0,
      0,
      "print the threadtree",
      0
   }, {
      0
   }
};

// Option parsing function
static error_t parse_cmd_options(int key, char *arg, struct argp_state *state) {
   // Get the input argument from argp_parse, which we
   // know is a pointer to our arguments structure. */
   cmd_options_t *options = state->input;

   switch (key) {
      case output_filename_ID:
         options->output_filename = arg;
         break;
      case only_header_ID:
         options->only_header = true;
         break;
      case skip_samples_ID:
         options->skip_samples = true;
         break;
      case show_threadtree_ID:
         options->show_threadtree = true;
         break;
      case ARGP_KEY_ARG:
         if (state->arg_num == 0) {
            options->vfd_filename = arg;
         } else {
            argp_usage(state);
            return ARGP_ERR_UNKNOWN;
         }
         break;
      case ARGP_KEY_ARGS:
         break;
      default:
         return ARGP_ERR_UNKNOWN;
         break;
   }

   return 0;
}

// out argp parser
static struct argp argp = {possible_options,
                           parse_cmd_options,
                           NULL,
                           doc,
                           NULL, NULL, NULL};

// easy interface to command line option parsing
cmd_options_t parse_command_line_options(int argc, char **argv) {
   cmd_options_t options;
   options.output_filename = "stdout";
   options.vfd_filename = NULL;
   options.only_header = false;
   options.skip_samples = false;
   options.show_threadtree = false;

   // actually parse the arguments
   argp_parse(&argp, argc, argv, 0, 0, &options);

   return options;
}
