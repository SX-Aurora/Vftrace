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
#include <ctype.h>

#include <math.h>

//#include "vftr_stacks.h"

#define LINEBUFSIZE 4096

int read_n_stacks (FILE *fp) {
   char buf[2*LINEBUFSIZE + 1];
   fseek (fp, -2*LINEBUFSIZE, SEEK_END);
   int size = fread (buf, sizeof(char), 2*LINEBUFSIZE, fp);
   int i = size;
   int decimal_place = 1;
   int n_stacks = 0;
   bool read_number = false;
   while (i >= 0) {
       if (!read_number) {
   	read_number = buf[i] == ' ';
       } else {
          if (isdigit(buf[i])) {
   	  n_stacks += (buf[i] - '0') * decimal_place;
   	  decimal_place *= 10;
          } else {
   	  break;
          }
       }
       i--;
   }
   rewind (fp);
   return n_stacks;
}

/**********************************************************************/

void decompose_table_line (char *line, int *n_calls, double *t_excl, double *t_incl, double *p_abs, double *p_cum,
			   char **func_name, char **caller_name, int *stack_id) {
   char *token;
   token = strtok (line, " ");
   *n_calls = atoi(token);
   token = strtok (NULL, " ");
   *t_excl = atof (token);
   token = strtok (NULL, " ");
   *t_incl = atof (token);
   token = strtok (NULL, " ");
   *p_abs = atof (token);
   token = strtok (NULL, " ");
   *p_cum = atof (token);
   *func_name = strtok (NULL, " ");
   *caller_name = strtok (NULL, " ");
   token = strtok (NULL, " ");	
   *stack_id = atoi(token);
}

/**********************************************************************/

void read_table (FILE *fp, double t[], int stack_pos[], char *func_names[]) {
   char line[LINEBUFSIZE];
   int countdown = -1;
   int i_t = 0;
   while (!feof(fp)) {		
      fgets (line, LINEBUFSIZE, fp);
      if (countdown < 0)  {
         if (strstr(line, "Runtime profile for rank")) countdown = 4;
      } else if (countdown > 0) {
         countdown--;
      } else {
         //printf ("Line: %s\n", line);
         if (strstr(line, "--------")) break;
         int n_calls, stack_id;
         double t_excl, t_incl, p_abs, p_cum;
         char *func_name, *caller_name; 
         decompose_table_line (line, &n_calls, &t_excl, &t_incl, &p_abs, &p_cum,
   			    &func_name, &caller_name, &stack_id);
         t[i_t] = t_excl;
	 func_names[stack_id] = strdup(func_name);
	 //if (i_t == 0) printf ("func_name: %s %s\n", func_name, func_names[i_t]);
         stack_pos[stack_id] = i_t;
         i_t++;
      }
   }
}

/**********************************************************************/

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


int main (int argc, char *argv[]) {

	if (argc < 3) {
		printf ("Need two arguments\n");
		return -1;
	}

	FILE *fp1 = fopen (argv[1], "r");
	FILE *fp2 = fopen (argv[2], "r");
	
	int n_stacks_1 = read_n_stacks (fp1);
	int n_stacks_2 = read_n_stacks (fp2);

	if (n_stacks_1 != n_stacks_2) {
	   printf ("Nr. of stacks do not match: %d %d\n", n_stacks_1, n_stacks_2);
	   return -1;
	}

	printf ("n_stacks in %s: %d\n", argv[1], n_stacks_1);
	printf ("n_stacks in %s: %d\n", argv[2], n_stacks_2);

	int stack_id_position_1[n_stacks_1];
	int stack_id_position_2[n_stacks_2];
	double t1[n_stacks_1];
	double t2[n_stacks_2];

	char *func_names_1[n_stacks_1];
	char *func_names_2[n_stacks_2];

	//double t_diff[n_stacks_1];
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

	read_table (fp1, t1, stack_id_position_1, func_names_1);
	read_table (fp2, t2, stack_id_position_2, func_names_2);
	//printf ("Check: %s %s\n", func_names_1[0], func_names_2[0]);

	for (int i = 0; i < n_stacks_1; i++) {
	   //t_diff[i] = t1[stack_id_position_1[i]] - t1[stack_id_position_2[i]];
	   int i1 = stack_id_position_1[i];
	   int i2 = stack_id_position_2[i];
	   deltas[i]->t_diff_abs = t1[i1] - t2[i2];
	   if (t1[i1] > 0.0) {
	      //deltas[i]->t_diff_rel = deltas[i]->t_diff_abs / t1[stack_id_position_1[i]];
	      deltas[i]->t_diff_rel = t1[stack_id_position_2[i]] / t2[stack_id_position_1[i]];
	   }
	   deltas[i]->stack_id = i;
	}

	qsort (deltas, (size_t)n_stacks_1, sizeof(double*), sort_by_absolute_time);

	for (int i = 0; i < n_stacks_1; i++) {
		int stack_id = deltas[i]->stack_id;
		printf ("%4d:   %lf   %lf   %7.3f   %7.3f   %s\n",
			 stack_id, t1[stack_id_position_1[stack_id]], t2[stack_id_position_2[stack_id]], 
			 deltas[i]->t_diff_abs, deltas[i]->t_diff_rel, func_names_1[stack_id]);
	}

	fclose (fp1);
	fclose (fp2);
	
	return 0;
}
