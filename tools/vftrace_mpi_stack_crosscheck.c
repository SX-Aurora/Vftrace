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
#include <stdbool.h>
#include <math.h>
#include <float.h>
#include <ctype.h>

#define N_MPI_FUNCS 13
#define LINEBUFSIZE 256
char *mpi_function_names[N_MPI_FUNCS] = {"mpi_barrier", "mpi_bcast", "mpi_reduce",
             		                 "mpi_allreduce", "mpi_gatherv", "mpi_gather",
             		                 "mpi_allgatherv", "mpi_allgather",
             		                 "mpi_scatterv", "mpi_scatter",
             		                 "mpi_alltoallv", "mpi_alltoallw", "mpi_alltoall"};

/**********************************************************************/

int mpi_index (char *s) {
	char *tmp = s;
	bool before = true;
        while (*tmp != '\0') {
	   if (!isalpha(*tmp) && !isdigit(*tmp) && *tmp != '_') {
	      if (before) {
	         s = tmp;
	      } else {
		 *tmp = '\0';
	      }
           } else {
	      before = false;
	   }
	   tmp++;
	}
	for (int i = 0; i < N_MPI_FUNCS; i++) {
		if (strstr (s, mpi_function_names[i])) {
			return i;
		}
	}
	return -1;
}

/**********************************************************************/

bool equal_for_n_digits (double val1, double val2, int n_digits) {
	double max_diff = 1.0;
	for (int i = 0; i < n_digits; i++) {
	   max_diff /= 10.0;
	}
	return (fabs(val1 - val2) <= max_diff);
}

bool equal_within_tolerance (double val1, double val2, double tolerance) {
	if (val2 == 0.0) {
	   return fabs(val1) < tolerance;
	} else {
	   return fabs (val1 / val2 - 1.0) < tolerance;
	}
}

/**********************************************************************/

int decompose_mpi_size (char *line) {
   char *token;
   token = strtok (line, " ");
   token = strtok (NULL, " ");
   token = strtok (NULL, " ");
   return atoi(token);
}

void decompose_stack_leaf_header (char *line, int *i_mpi, int *n_stacks) {
   char *token;
   int i = 0;
   token = strtok (line, " ");
   while (i++ < 4) token = strtok (NULL, " ");
   *i_mpi = mpi_index(token);
   token = strtok (NULL, " ");
   if (!token) {
      *n_stacks = 0;
   } else {
      *n_stacks = atoi(token);
   }
}

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

bool decompose_stack_line (char *line, double *t, int *n_calls, double *imba, int *stack_id) {
   char *token;
   if (strstr(line, "not on this rank")) return false;
   token = strtok (line, " ");
   token = strtok (NULL, " ");
   *t = atof(token);
   token = strtok (NULL, " ");
   *n_calls = atoi(token);
   token = strtok (NULL, " ");
   *imba = atof(token);
   token = strtok (NULL, " ");
   *stack_id = atoi(token);
   return true;
}

void decompose_final_line (char *line, int *i_mpi, double *t_tot) {
   char *token;
   token = strtok (line, " ");
   // Skip the six characters of Total(
   token += 6;
   // Loop to the closing bracket and set the terminator there.
   char *tmp = token;
   while (*tmp != ')') tmp++;
   *tmp = '\0';
   *i_mpi = mpi_index (token);
   token = strtok (NULL, " ");
   *t_tot = atof (token);
}

/**********************************************************************/

int find_stack_index (int *all_ids[N_MPI_FUNCS][64], int i_mpi, int i_file, int search_id, int n_stacks) {
  int i_stack;
  for (i_stack = 0; i_stack < n_stacks; i_stack++) {
     if (all_ids[i_mpi][i_file][i_stack] == search_id) return i_stack;
  }
  return -1;
}

/**********************************************************************/

int main (int argc, char *argv[]) {
   int n_log_files = argc - 1;
   int n_stacks[N_MPI_FUNCS][n_log_files];
   long filepos[N_MPI_FUNCS][n_log_files];
   int *all_n_calls[N_MPI_FUNCS][n_log_files];
   double *all_t[N_MPI_FUNCS][n_log_files];
   double *all_imba[N_MPI_FUNCS][n_log_files];
   int *all_stack_ids[N_MPI_FUNCS][n_log_files];
   int *all_stack_ids_for_i_mpi[N_MPI_FUNCS];
   int n_stack_ids[N_MPI_FUNCS];
   int mpi_size = 0;
   bool all_okay;
   double t_tot[N_MPI_FUNCS][n_log_files];
   for (int i = 1; i < argc; i++) {
      FILE *fp = fopen (argv[i], "r");
      int n_mpi_read = 0;
      if (!fp) {
         printf ("Error: Could not open %s\n", argv[i]);
	 return -1;
      }
      int i_file = i - 1;
      char line[LINEBUFSIZE];
      int i_mpi;
      // First pass:  Count stacks and register the file positions
      while (!feof(fp)) {
	 int n;
	 long this_fp = ftell(fp);
         fgets (line, LINEBUFSIZE, fp);
	 bool has_subsequent_leaves = strstr(line, "Function stacks leading to");
         bool has_no_stack_ids = strstr(line, "No stack IDs");
	 if (strstr (line, "MPI size")) {
	    mpi_size = decompose_mpi_size (line);
	 } else if (has_subsequent_leaves || has_no_stack_ids) {
	    decompose_stack_leaf_header (line, &i_mpi, &n);
	    n_stacks[i_mpi][i_file] = n;
	    filepos[i_mpi][i_file] = this_fp;
	    n_mpi_read++;
         }
         if (has_subsequent_leaves) {
            all_n_calls[i_mpi][i_file] = (int*) malloc (n_stacks[i_mpi][i_file] * sizeof(int));
	    all_t[i_mpi][i_file] = (double*) malloc (n_stacks[i_mpi][i_file] * sizeof(double));
	    all_imba[i_mpi][i_file] = (double*) malloc (n_stacks[i_mpi][i_file] * sizeof(double));
	    all_stack_ids[i_mpi][i_file] = (int*) malloc (n_stacks[i_mpi][i_file] * sizeof(int));
	    for (int i = 0; i < n_stacks[i_mpi][i_file]; i++) {
	       all_n_calls[i_mpi][i_file][i] = 0;
	       all_t[i_mpi][i_file][i] = 0.0;
	       all_imba[i_mpi][i_file][i] = 0.0;
	       all_stack_ids[i_mpi][i_file][i] = -1;
	    }
        }
        if (n_mpi_read == N_MPI_FUNCS) break;
      }
      long current_filepos = 0;
      int *this_stack_ids[N_MPI_FUNCS];
      int i_stack[N_MPI_FUNCS];
      for (i_mpi = 0; i_mpi < N_MPI_FUNCS; i_mpi++) {
         this_stack_ids[i_mpi] = (int*)malloc (n_stacks[i_mpi][i_file] * sizeof(int));
	 i_stack[i_mpi] = 0;
      }

      rewind (fp);
      for (i_mpi = 0; i_mpi < N_MPI_FUNCS; i_mpi++) {
        fseek (fp, filepos[i_mpi][i_file] - current_filepos, SEEK_CUR);
        fgets (line, LINEBUFSIZE, fp);
        fgets (line, LINEBUFSIZE, fp);
        for (int i_stack = 0; i_stack < n_stacks[i_mpi][i_file]; i_stack++) {
            fgets (line, LINEBUFSIZE, fp);
	    double t, imba;
	    int n_calls, stack_id;
       	    if (decompose_stack_line (line, &t, &n_calls, &imba, &stack_id)) {
	       all_t[i_mpi][i_file][i_stack] = t;
	       all_n_calls[i_mpi][i_file][i_stack] = n_calls;
	       all_imba[i_mpi][i_file][i_stack] = imba;
	       all_stack_ids[i_mpi][i_file][i_stack] = stack_id;
	    }
        }
        fgets (line, LINEBUFSIZE, fp);
	if (n_stacks[i_mpi][i_file] > 0) {
	 	int dummy;
		decompose_final_line (line, &dummy, &(t_tot[i_mpi][i_file]));
	}
        current_filepos = ftell(fp);
      }
   }

   printf ("Check that the summed up times in each file match with the preceeding times:\n");
   all_okay = true;
   for (int i_file = 0; i_file < n_log_files; i_file++) {
     for (int i_mpi = 0; i_mpi < N_MPI_FUNCS; i_mpi++) {
        double t_sum = 0;
	for (int i_stack = 0; i_stack < n_stacks[i_mpi][i_file]; i_stack++) {
	   t_sum += all_t[i_mpi][i_file][i_stack];
	}
	if (!equal_within_tolerance (t_sum, t_tot[i_mpi][i_file], 0.02)) {
	   printf ("Not okay: %d %d %lf %lf\n", i_file, i_mpi, t_sum, t_tot[i_mpi][i_file]);
	   all_okay = false;
	}
     }
   }
   if (all_okay) printf ("ALL OKAY\n");

   if (n_log_files != mpi_size) {
     printf ("The number of log files does not match the registered MPI size! ");
     printf ("Found %d log files, but there should be %d.\n", n_log_files, mpi_size);
     return -1;
   }

   printf ("Check that number of stacks matches arcoss all ranks\n");
   for (int i_mpi = 0; i_mpi < N_MPI_FUNCS; i_mpi++) {
      int n_stacks_0 = n_stacks[i_mpi][0];
      all_okay = true;
      for (int i_file = 1; i_file < n_log_files; i_file++) {
	 all_okay &= n_stacks[i_mpi][i_file] == n_stacks_0;
      }
      printf ("%d: %s\n", i_mpi, all_okay ? "OKAY" : "NOT OKAY");
      n_stack_ids[i_mpi] = n_stacks_0;
      all_stack_ids_for_i_mpi[i_mpi] = (int*) malloc (n_stacks_0 * sizeof(int));
      for (int i_stack = 0; i_stack < n_stacks_0; i_stack++) {
	 all_stack_ids_for_i_mpi[i_mpi][i_stack] = -1;
      }
      int n_found = 0;
      for (int i_file = 0; i_file < n_log_files; i_file++) {
         for (int i_stack = 0; i_stack < n_stacks_0; i_stack++) {
  	    if (n_found == 0) {
		all_stack_ids_for_i_mpi[i_mpi][n_found] = all_stack_ids[i_mpi][i_file][i_stack];
	        n_found++;
	    } else {
		bool found = false;
	        for (int ii = 0; ii < n_found; ii++) {
	 	   found |= all_stack_ids[i_mpi][i_file][i_stack] == all_stack_ids_for_i_mpi[i_mpi][ii];
	        }
		if (!found) all_stack_ids_for_i_mpi[i_mpi][n_found++] = all_stack_ids[i_mpi][i_file][i_stack];
	    }
         }
      }
   }

   printf ("Check imbalances:\n");
   double *t_avg[N_MPI_FUNCS];
   double *max_diff[N_MPI_FUNCS];
   for (int i_mpi = 0; i_mpi < N_MPI_FUNCS; i_mpi++) {
      int this_n_stacks = n_stack_ids[i_mpi];
      t_avg[i_mpi] = (double*) malloc (this_n_stacks * sizeof(double));
      max_diff[i_mpi] = (double*) malloc (this_n_stacks * sizeof(double));
      for (int i_stack = 0; i_stack < this_n_stacks; i_stack++) {
         t_avg[i_mpi][i_stack] = 0.0;
	 max_diff[i_mpi][i_stack] = 0.0;
      }
      for (int i_stack = 0; i_stack < this_n_stacks; i_stack++) {
	 int n = 0;
	 int this_stack_id = all_stack_ids_for_i_mpi[i_mpi][i_stack];
         for (int i_file = 0; i_file < n_log_files; i_file++) {
	    int this_i_stack = find_stack_index (all_stack_ids, i_mpi, i_file, this_stack_id, this_n_stacks);
	    if (all_t[i_mpi][i_file][this_i_stack] > 0.0) {
               t_avg[i_mpi][i_stack] += all_t[i_mpi][i_file][this_i_stack];
	       n++;
            }
         }
	 if (n > 0) t_avg[i_mpi][i_stack] /= n;
	 for (int i_file = 0; i_file < n_log_files; i_file++) {
	    int this_i_stack = find_stack_index (all_stack_ids, i_mpi, i_file, this_stack_id, this_n_stacks);
	    if (all_t[i_mpi][i_file][this_i_stack] > 0.0) {
	       double d = fabs (all_t[i_mpi][i_file][this_i_stack] - t_avg[i_mpi][i_stack]);
	       if (d > max_diff[i_mpi][i_stack]) {
		 max_diff[i_mpi][i_stack] = d;
	       }
            }
	 }
         bool all_okay = true;
	 for (int i_file = 0; i_file < n_log_files; i_file++) {
	    int this_i_stack = find_stack_index (all_stack_ids, i_mpi, i_file, this_stack_id, this_n_stacks);
	    if (this_i_stack < 0) continue;
	    if (all_t[i_mpi][i_file][this_i_stack] == 0.0) continue;
	    all_okay = all_okay && equal_within_tolerance (all_imba[i_mpi][i_file][this_i_stack], max_diff[i_mpi][i_stack] / t_avg[i_mpi][i_stack] * 100, 0.05);
	    }
	 printf ("%s(%d): %s\n", mpi_function_names[i_mpi], i_stack, all_okay ? "OKAY" : "NOT OKAY");
	 }
      }
   return 0;
}
