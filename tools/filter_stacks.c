/*
   This file is part of Vftrace.

   Vftrace is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.

   Vftrace is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License along
   with this program; if not, write to the Free Software Foundation, Inc.,
   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <argp.h>

#include "vftr_stacks.h"

#define LINEBUFSIZE 256

struct arguments {
   double t_min;
   int n_calls_min;
   double imba_min;
   char *filename;
};

static struct argp_option options[] = {
   {"min-imbalance", 'i', "IMBA_MIN", 0, "Minimum imbalance on stack branch in percent."},
   {"min-calls", 'n', "N_CALLS_MIN", 0, "Minimum number of calls."},
   {"min-time", 't', "T_MIN", 0, "Minimum time spent in stack branch."},
   {0}
};

static error_t parse_opt (int key, char *arg, struct argp_state *state) {
   struct arguments *arguments = state->input;
   
   switch(key) {
      case 'i':
        arguments->imba_min = atof(arg);
        break;
      case 'n': 
        arguments->n_calls_min = atoi(arg);
	break;
      case 't':
        arguments->t_min = atof(arg);
        break;
      case ARGP_KEY_ARG:
        arguments->filename = arg;
	break;
      case ARGP_KEY_END:
        if (state->arg_num < 1) argp_usage (state);
        break;
      default:
        return ARGP_ERR_UNKNOWN;
    }
    return 0;
}

static char args_doc[] = "FOO";
static char doc[] = "BAR";
static struct argp argp = {options, parse_opt, args_doc, doc};

/**********************************************************************/

void read_header (char *line, char **func_name, int *n_funcs) {
   // Header starts with "Function stacks leading to <func>: n_funcs
   // First, we skip 4 tokens.
   int i = 0;
   char *token = strtok (line, " ");
   while (i++ < 4) token = strtok (NULL, " ");
   // This is the function name. We scan the token for a possible ':' at the end and
   // remove it.
   char *tmp = token;
   while (*tmp != ':' && *tmp != '\0') tmp++;
   if (*tmp == ':') *tmp = '\0';
   *func_name = token;
   // Next is the number of functions which follow.
   token = strtok (NULL, " ");
   *n_funcs = atoi (token);
}

void read_stack_line (char *line, int *n_spaces, char **branch,
		      double *this_t, int *n_calls, double *imba) {
   // First, we count the number of spaces before the branch token.
   *n_spaces = 0;
   while (*line == ' ') {
      (*n_spaces)++;
      line++;
   }
   // Next, we split up the line, obtaining first the branch string and then the values
   // specified in the interface above.
   *branch = strtok (line, " ");
   *this_t = atof (strtok (NULL, " "));
   *n_calls = atoi (strtok (NULL, " "));
   *imba = atof (strtok (NULL, " "));
   
}

/**********************************************************************/

#define STACK_STRING_SIZE 100
#define TIME_COLUMN_SIZE 13
#define CALL_COLUMN_SIZE 8
#define IMBA_COLUMN_SIZE 6

void get_formats (int *fmt_t, int *fmt_n_calls, int *fmt_imba) {
  *fmt_t = TIME_COLUMN_SIZE > strlen(vftr_stacktree_headers[TIME]) ? TIME_COLUMN_SIZE : strlen(vftr_stacktree_headers[TIME]);
  *fmt_n_calls = CALL_COLUMN_SIZE > strlen(vftr_stacktree_headers[CALLS]) ? CALL_COLUMN_SIZE : strlen(vftr_stacktree_headers[CALLS]);
  *fmt_imba = IMBA_COLUMN_SIZE > strlen(vftr_stacktree_headers[IMBA]) ? IMBA_COLUMN_SIZE : strlen(vftr_stacktree_headers[IMBA]);
}

/**********************************************************************/

void print_dashes (int n) {
   for (int i = 0; i < n; i++) printf ("-");
   printf ("\n");
}

/**********************************************************************/

void print_table_header (struct arguments arguments, char *func_name, int fmt_t, int fmt_n_calls, int fmt_imba) {
   char table_header[STACK_STRING_SIZE + 1];
   for (int i = 0; i <= STACK_STRING_SIZE; i++) {
      table_header[i] = (i == STACK_STRING_SIZE) ? '\0' : ' ';
   }
   char *tmp = &table_header[0];
   int nw = snprintf (tmp, STACK_STRING_SIZE, "Call paths for %s ", func_name);
   tmp += nw;
   if (arguments.t_min > 0) {
      nw = snprintf (tmp, STACK_STRING_SIZE, "(T[s] > %lf) ", arguments.t_min);
      tmp += nw;
   }
   if (arguments.n_calls_min > 0) {
      nw = snprintf (tmp, STACK_STRING_SIZE, "(n_calls > %d) ", arguments.n_calls_min);
      tmp += nw;
   }
   if (arguments.imba_min > 0) {
      nw = snprintf (tmp, STACK_STRING_SIZE, "(imbalance > %lf %%) ", arguments.imba_min);
      tmp += nw;
   }
   snprintf (tmp++, STACK_STRING_SIZE, ":");
   *tmp = ' '; 

   printf ("\n");
   printf ("%s %*s %*s %*s\n", table_header, fmt_t, vftr_stacktree_headers[TIME],
 	  fmt_n_calls, vftr_stacktree_headers[CALLS], fmt_imba, vftr_stacktree_headers[IMBA]);
   print_dashes (STACK_STRING_SIZE + fmt_t + fmt_n_calls + fmt_imba + 3);
}

/**********************************************************************/

char stack_string[STACK_STRING_SIZE];

void overwrite_stack_string (char *new, int n_spaces) {
    int i = 0;
    if (stack_string[0] == '\0') {
       // Don't forget that strlen does not count the terminating character.
       for (i = 0; i < strlen(new); i++) {
           stack_string[i] = new[i];
        }
       stack_string[i] = '\0';
    } else {
       char *tmp = &stack_string[0];
       while (i++ <= n_spaces) tmp++;
       i = 0;
       while (i++ <= strlen(new)) {
          *tmp = new[i];
          tmp++;
       }
       *tmp = '\0';
    }
}

/**********************************************************************/

bool print_this_line (struct arguments arguments, double t, int n_calls, double imba) {
   return t >= arguments.t_min && n_calls >= arguments.n_calls_min && imba >= arguments.imba_min;
}

/**********************************************************************/

int main (int argc, char *argv[]) {

   //char *filename = argv[1];
   //
   struct arguments arguments;
   arguments.t_min = 0.0;
   arguments.n_calls_min = 0;
   arguments.imba_min = 0.0;
   arguments.filename = NULL;

   argp_parse (&argp, argc, argv, 0, 0, &arguments);

   FILE *fp;
   if (!(fp = fopen (arguments.filename, "r"))) {
	printf ("Could not open %s!\n", arguments.filename);
	return -1;
   }

   char line[LINEBUFSIZE];
   char *func_name;
   int n_spaces, n_calls;
   int n_funcs = 0;
   double this_t, imba;
   int fmt_t, fmt_n_calls, fmt_imba;
   get_formats (&fmt_t, &fmt_n_calls, &fmt_imba);
   while (!feof(fp)) {
      if (n_funcs > 0) break;
      fgets (line, LINEBUFSIZE, fp);
      if (strstr (line, "Function stacks")) {
	  int n_filtered = 0;
          read_header (line, &func_name, &n_funcs);
	  print_table_header (arguments, func_name, fmt_t, fmt_n_calls, fmt_imba);
	  stack_string[0] = '\0';
	  for (int i = 0; i < n_funcs + 1; i++) {
	      fgets (line, LINEBUFSIZE, fp);
	      // The first line is the delimiter "----".
 	      if (i == 0) continue;
	      char *branch;
	      read_stack_line (line, &n_spaces, &branch, &this_t, &n_calls, &imba);
	      overwrite_stack_string (branch, n_spaces);
	      if (print_this_line (arguments, this_t, n_calls, imba)) {
		 printf ("%*s %*.6f %*d %*.2lf\n", STACK_STRING_SIZE, stack_string,
		         fmt_t, this_t, fmt_n_calls, n_calls, fmt_imba, imba);
		 n_filtered++;
	      }
	  }
	  if (n_filtered == 0) {
	     printf ("%*s\n", STACK_STRING_SIZE, "NONE");
	  }
	  print_dashes (STACK_STRING_SIZE + fmt_t + fmt_n_calls + fmt_imba + 3);
      }
   }

   fclose(fp);
   return 0;
}
