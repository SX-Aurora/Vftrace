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
#include <math.h>

#include "regular_expressions.h"

#define LINEBUFSIZE 4096

// We loop through the file and search for lines starting with "  STID[0-9]".
// Each one of them indicates a stack tree element.
int read_n_stacks (FILE *fp) {
   char buf[LINEBUFSIZE];
   int n_stacks = 0;
   regex_t *r = vftr_compile_regexp("^[ \t]+STID[0-9]*");
   
   while (fgets(buf, LINEBUFSIZE, fp)) {
      if (vftr_pattern_match(r, buf)) n_stacks++;
   }
   rewind (fp);
   return n_stacks;
}

/**********************************************************************/

// The standard table columns are | Calls | t_excl | t_incl | Function | Caller | ID |
void decompose_table_line (char *line, int *n_calls, double *t_excl, double *t_incl,
			   char **func_name, int *stack_id) {
   char *token;
   token = strtok (line, "|");
   *n_calls = atoi(token);
   token = strtok (NULL, "|");
   *t_excl = atof (token);
   token = strtok (NULL, "|");
   *t_incl = atof (token);
   // Skip all the spaces in front of the function name
   token = strtok (NULL, "|");
   *func_name = strtok (NULL, " ");
   // Caller name, unused in this program
   char *caller_name = strtok (NULL, "|");
   token = strtok (NULL, "|");
   *stack_id = atoi(token);
}

/**********************************************************************/

// Helper functions for string formatting
int count_digits_long (long long value) {
  if (value == 0) {
     return 1;
  } else {
     int count = 0;
     for (int c = value; c > 0; c /= 10) {
           count++;
     }
     return count;
  }
}

int count_digits_int (int value) {
  return count_digits_long ((long long )value);
}

int count_digits_double (double value) {
  return count_digits_long ((long long)floor(value));
}

/**********************************************************************/

void read_table (FILE *fp, double t[], int stack_pos[], char *func_names[]) {
   char line[LINEBUFSIZE];
   int countdown = -1;
   int i_t = 0;
   while (fgets(line, LINEBUFSIZE, fp)) {
      if (countdown < 0)  {
         // Do not match the "Runtime profile for application: " in the header
         if (!strcmp(line, "Runtime profile\n")) countdown = 3;
      } else if (countdown > 0) {
         countdown--;
      } else {
         // End of profile table. Finish parsing.
         if (strstr(line, "--------")) break;
         int n_calls, stack_id;
         double t_excl, t_incl;
         char *func_name;
         decompose_table_line (line, &n_calls, &t_excl, &t_incl, &func_name, &stack_id);
         t[i_t] = t_incl;
	 func_names[stack_id] = strdup(func_name);
         stack_pos[stack_id] = i_t;
         i_t++;
      }
   }
}

/**********************************************************************/

// Store the difference of a function pair, identified by its stack ID.
typedef struct delta {
	double t_diff_abs;
	double t_diff_rel;
	int stack_id;
} delta_t;

int sort_by_absolute_time (const void *a1, const void *a2) {
    delta_t *d1 = *(delta_t **)a1;
    delta_t *d2 = *(delta_t **)a2;
    double fdiff = fabs(d2->t_diff_abs) - fabs(d1->t_diff_abs);
    if (fdiff > 0) return 1;
    if (fdiff < 0) return -1;
    return 0;
}


int sort_by_relative_time (const void *a1, const void *a2) {
    delta_t *d1 = *(delta_t **)a1;
    delta_t *d2 = *(delta_t **)a2;
    double fdiff = fabs(d2->t_diff_rel) - fabs(d1->t_diff_rel);
    if (fdiff > 0) return 1;
    if (fdiff < 0) return -1;
    return 0;
}

/**********************************************************************/

void display_table (int n_stacks, delta_t **deltas, double *t1, double *t2,
                    int *stack_id_position_1, int *stack_id_position_2, char **func_names) {
        // Determine maximal string length for table formatting
        int smax = 0;
        for (int i = 0; i < n_stacks; i++) {
           int stack_id = deltas[i]->stack_id;
           int slen = strlen(func_names[stack_id]);
           if (slen > smax) smax = slen;
        }
        int nmax_t1 = strlen("T1[s]");
        int nmax_t2 = strlen("T2[s]");
        int nmax_tdiff = strlen("T_diff[s]");
        int nmax_tdiff_rel = nmax_tdiff; 
        int nmax_stack_id = strlen("stackID");
        for (int i = 0; i < n_stacks; i++) {
           int stack_id = deltas[i]->stack_id;
           int i_t1 = stack_id_position_1[stack_id];
           int i_t2 = stack_id_position_2[stack_id];
           if (i_t1 > 0) {
              int n = count_digits_double (t1[i_t1]) + 3;
              if (n > nmax_t1) nmax_t1 = n;
           }  
           if (i_t2 > 0) {
              int n = count_digits_double (t2[i_t2]) + 3;
              if (n > nmax_t2) nmax_t2 = n;
           } 
           int ndiff = count_digits_double(deltas[i]->t_diff_abs) + 3;
           if (ndiff > nmax_tdiff) nmax_tdiff = ndiff;
        }


	printf ("%*s T1[s] T2[s] T_diff[s] T_diff[%] stackID\n", smax, "Function");
	for (int i = 0; i < n_stacks; i++) {
		int stack_id = deltas[i]->stack_id;
		int i_t1 = stack_id_position_1[stack_id];
		int i_t2 = stack_id_position_2[stack_id];
		if (i_t1 > 0 && i_t2 > 0) {
		   printf ("%*s %*.2f %*.2f %*.2f %*.2f %*d\n", 
                           smax, func_names[stack_id],
                           nmax_t1, t1[i_t1], nmax_t2, t2[i_t2],
                           nmax_tdiff, deltas[i]->t_diff_abs,
                           nmax_tdiff_rel, deltas[i]->t_diff_rel,
                           nmax_stack_id, stack_id);
		}
	}
}

/**********************************************************************/

int main (int argc, char *argv[]) {

	if (argc < 3) {
		printf ("Need two Vftrace profile files to compare as input!\n");
		return -1;
	}

	FILE *fp1 = fopen (argv[1], "r");
	FILE *fp2 = fopen (argv[2], "r");

	int n_stacks_1 = read_n_stacks (fp1);
        int n_stacks_2 = read_n_stacks (fp2);

	if (n_stacks_1 != n_stacks_2) {
	   printf ("Nr. of stacks does not match: %d %d\n", n_stacks_1, n_stacks_2);
	   return -1;
	}

	int stack_id_position_1[n_stacks_1];
	int stack_id_position_2[n_stacks_2];
	double t1[n_stacks_1];
	double t2[n_stacks_2];

	char *func_names_1[n_stacks_1];
	char *func_names_2[n_stacks_2];

	delta_t **deltas = (delta_t**) malloc (n_stacks_1 * sizeof(delta_t*));

	for (int i = 0; i < n_stacks_1; i++) {
	   stack_id_position_1[i] = -1;
	   stack_id_position_2[i] = -1;
	   t1[i] = 0.0;
	   t2[i] = 0.0;
	   deltas[i] = (delta_t*) malloc (sizeof(delta_t));
	   deltas[i]->t_diff_abs = 0.0;
	   deltas[i]->t_diff_rel = 0.0;
	   deltas[i]->stack_id = -1;
	}

        // In the profile table, functions with the same stack ID do not need
        // to be on the same position. The arrays stack_id_position indicate
        // at which position in the profile table a given stack id can be found.
	read_table (fp1, t1, stack_id_position_1, func_names_1);
	read_table (fp2, t2, stack_id_position_2, func_names_2);
	
	// Logfiles aren't needed any more.
	fclose (fp1);
	fclose (fp2);

	for (int i = 0; i < n_stacks_1; i++) {
	   int i1 = stack_id_position_1[i];
	   int i2 = stack_id_position_2[i];
	   deltas[i]->t_diff_abs = t1[i1] - t2[i2];
	   if (t1[i1] > 0.0) {
	      deltas[i]->t_diff_rel = deltas[i]->t_diff_abs / t1[stack_id_position_1[i]] * 100;
	   }
	   deltas[i]->stack_id = i;
	}

	qsort (deltas, (size_t)n_stacks_1, sizeof(double*), sort_by_absolute_time);
        display_table (n_stacks_1, deltas, t1, t2, stack_id_position_1, stack_id_position_2, func_names_1);

	return 0;
}
