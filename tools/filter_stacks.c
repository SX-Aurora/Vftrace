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
#include <string.h>

#include "vftr_stacks.h"

#define LINEBUFSIZE 256

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

int main (int argc, char *argv[]) {

   char *filename = argv[1];
   FILE *fp;
   if (!(fp = fopen (argv[1], "r"))) {
	printf ("Could not open %s!\n", argv[1]);
	return -1;
   }

   double t_threshold = 0.01;

   char line[LINEBUFSIZE];
   char *func_name;
   int n_spaces, n_calls;
   int n_funcs = 0;
   double this_t, imba;
   while (!feof(fp)) {
      if (n_funcs > 0) break;
      fgets (line, LINEBUFSIZE, fp);
      if (strstr (line, "Function stacks")) {
          read_header (line, &func_name, &n_funcs);
	  //printf ("Function: %s, n: %d\n", func_name, n_funcs);
	  char tot_function_string[LINEBUFSIZE];
	  for (int i = 0; i < n_funcs + 1; i++) {
	      fgets (line, LINEBUFSIZE, fp);
 	      if (i == 0) continue;
	      char *branch;
	      read_stack_line (line, &n_spaces, &branch, &this_t, &n_calls, &imba);
 	      //printf ("n_spaces: %d, branch: %s, this_t: %lf, n_calls: %d, imbal: %lf\n",
	      //	 n_spaces, branch, this_t, n_calls, imba);
	      int j;
	      if (i == 1) {
	         for (j = 0; j < strlen(branch); j++) {
	 	    tot_function_string[j] = branch[j];
	 	 }
		 tot_function_string[j] = '\0';
 	      } else {
		 char *tmp = &tot_function_string[0];
		 j = 0;
	         while (j++ <= n_spaces) tmp++;
		 //printf ("tmp: %s\n", tmp);
		 j = 0;
		 while (j++ <= strlen(branch)) {
		    *tmp = branch[j];
		    tmp++;
		 }
		 *tmp = '\0';
		 //printf ("tot_function_string: %s\n", tot_function_string);
	      }
	      if (this_t >= t_threshold) {
		 printf ("%s %lf %d %lf\n", tot_function_string, this_t, n_calls, imba);
	      }   
	  }
      }
   }

   fclose(fp);
   return 0;
}
