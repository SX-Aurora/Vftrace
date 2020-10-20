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

#define N_MPI_FUNCS 13
#define LINEBUFSIZE 256
char *mpi_function_names[N_MPI_FUNCS] = {"mpi_barrier", "mpi_bcast", "mpi_reduce",
             		                 "mpi_allreduce", "mpi_gatherv", "mpi_gather",
             		                 "mpi_allgatherv", "mpi_allgather",
             		                 "mpi_scatterv", "mpi_scatter",
             		                 "mpi_alltoallv", "mpi_alltoallw", "mpi_alltoall"};

int mpi_index (char *s) {
	for (int i = 0; i < N_MPI_FUNCS; i++) {
		if (strstr (s, mpi_function_names[i])) return i;
	}
	return -1;
}

bool equal_for_n_digits (double val1, double val2, int n_digits) {
	double max_diff = 1.0;
	for (int i = 0; i < n_digits; i++) {
	   max_diff /= 10.0;
	}
	return (fabs(val1 - val2) <= max_diff);
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

int main (int argc, char *argv[]) {
   int n_log_files = argc - 1;
   int n_stacks[N_MPI_FUNCS][n_log_files];
   long filepos[N_MPI_FUNCS][n_log_files];
   long table_filepos[N_MPI_FUNCS];
   int *all_n_calls[N_MPI_FUNCS][n_log_files];
   double *all_t[N_MPI_FUNCS][n_log_files];
   double *all_imba[N_MPI_FUNCS][n_log_files];
   int *all_stack_ids[N_MPI_FUNCS][n_log_files];
   int *all_stack_ids_for_i_mpi[N_MPI_FUNCS];
   int n_stack_ids[N_MPI_FUNCS];
   for (int i = 1; i < argc; i++) {
      FILE *fp = fopen (argv[i], "r");
      int max_n_stacks = 0;
      int n_mpi_read = 0;
      if (!fp) {
         printf ("Error: Could not open %s\n", argv[i]);
	 return -1;
      }
      int i_file = i - 1;
      char line[LINEBUFSIZE];
      char *token;
      int i_mpi;
      // First pass:  Count stacks and register the file positions 
      //printf ("First pass\n");
      while (!feof(fp)) {
	 long this_fp = ftell(fp);
         fgets (line, LINEBUFSIZE, fp);
         if (strstr (line, "Function stacks leading to")) {
            token = strtok (line, " "); 
	    int i = 0;
	    while (i++ < 4) token = strtok (NULL, " ");
	    i_mpi = mpi_index(token); 
	    token = strtok (NULL, " ");
	    n_stacks[i_mpi][i_file] = atoi (token);
            if (n_stacks[i_mpi][i_file] > max_n_stacks) max_n_stacks = n_stacks[i_mpi][i_file];
	    //printf ("this_fp: %ld\n", this_fp);
	    filepos[i_mpi][i_file] = this_fp;
            all_n_calls[i_mpi][i_file] = (int*) malloc (n_stacks[i_mpi][i_file] * sizeof(int));
	    all_t[i_mpi][i_file] = (double*) malloc (n_stacks[i_mpi][i_file] * sizeof(double));
	    all_imba[i_mpi][i_file] = (double*) malloc (n_stacks[i_mpi][i_file] * sizeof(double));
	    all_stack_ids[i_mpi][i_file] = (int*) malloc (n_stacks[i_mpi][i_file] * sizeof(int));
	    n_mpi_read++;
        } else if (strstr (line, "No stack IDs")) {
	    token = strtok (line, " ");
	    int i = 0; 
	    while (i++ < 4) token = strtok (NULL, " ");
	    i_mpi = mpi_index(token);
	    //printf ("No stack ids for: %s %d\n", token, i_mpi);
	    n_stacks[i_mpi][i_file] = 0;
	    filepos[i_mpi][i_file] = this_fp;
	    n_mpi_read++;
        }
        if (n_mpi_read == N_MPI_FUNCS) break;
      }   
      //if (n_mpi_read != N_MPI_FUNCS) {
      //   printf ("Not all MPI functions have been found in %s\n", argv[i]);
      //}
      long current_filepos = 0;
      int countdown_for_table = -1;
      int *this_stack_ids[N_MPI_FUNCS];
      int i_stack[N_MPI_FUNCS];
      for (i_mpi = 0; i_mpi < N_MPI_FUNCS; i_mpi++) {
         this_stack_ids[i_mpi] = (int*)malloc (n_stacks[i_mpi][i_file] * sizeof(int));
	 i_stack[i_mpi] = 0;
      }
      // Second pass: Register table stack IDs
      //printf ("Second pass\n");
      //rewind (fp);
      //while (!feof(fp)) {
      //   long this_fp = ftell(fp);
      //   fgets (line, LINEBUFSIZE, fp);
      //   if (strstr (line, "Runtime profile for rank")) {
      //      countdown_for_table = 4;
      //   } else if (countdown_for_table > 0) {
      //      countdown_for_table--;
      //   } else if (countdown_for_table == 0) {
      //      if (strstr(line, "-----")) break;
      //      int n_calls, stack_id;
      //      double t_excl, t_incl, p_abs, p_cum;
      //      char *func_name, *caller_name;
      //  	//printf ("Decompose: %s\n", line);
      //      decompose_table_line (line, &n_calls, &t_excl, &t_incl, &p_abs, &p_cum,
      //  		 	  &func_name, &caller_name, &stack_id);
      //  	//printf ("Decompose DONE\n");
      //      int i_mpi = mpi_index(func_name);
      //      if (i_mpi >= 0) {
      //         this_stack_ids[i_mpi][i_stack[i_mpi]++] = stack_id;
      //      }
      //   }
      //}

      //for (i_mpi = 0; i_mpi < N_MPI_FUNCS; i_mpi++) {
	 //printf ("Stack IDs for %s: \n", mpi_function_names[i_mpi]);
	 //for (int ii_stack = 0; ii_stack < n_stacks[i_mpi][i_file]; ii_stack++) {
	 //   printf ("%d ", this_stack_ids[i_mpi][ii_stack]);
	 //}
	 //printf ("\n");
      //}
      // Third pass: Parse stack trees and crosscheck #stacks

      //printf ("Third pass\n");
      rewind (fp);
      for (i_mpi = 0; i_mpi < N_MPI_FUNCS; i_mpi++) {
        //if (n_stacks[i_mpi][i_file] == 0) continue;
	//printf ("filepos: %ld\n", filepos[i_mpi][i_file]);
        fseek (fp, filepos[i_mpi][i_file] - current_filepos, SEEK_CUR);
        fgets (line, LINEBUFSIZE, fp);
	//printf ("line: %s\n", line);
        fgets (line, LINEBUFSIZE, fp);
	//printf ("line: %s\n", line);
        for (int i_stack = 0; i_stack < n_stacks[i_mpi][i_file]; i_stack++) {
            fgets (line, LINEBUFSIZE, fp);
	    double t, imba;
	    int n_calls, stack_id;
	    //printf ("decompose stack line: %s\n", line);
       	    if (decompose_stack_line (line, &t, &n_calls, &imba, &stack_id)) {
	       all_t[i_mpi][i_file][i_stack] = t;
	       all_n_calls[i_mpi][i_file][i_stack] = n_calls;
	       all_imba[i_mpi][i_file][i_stack] = imba;
	       all_stack_ids[i_mpi][i_file][i_stack] = stack_id;
	    }
        }	
        fgets (line, LINEBUFSIZE, fp);
        //printf ("Check: %s\n", line);
        current_filepos = ftell(fp);
      }

   }

   printf ("Check that #stacks match arcoss all ranks\n");
   for (int i_mpi = 0; i_mpi < N_MPI_FUNCS; i_mpi++) {
      int n_stacks_0 = n_stacks[i_mpi][0];
      bool all_okay = true;
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
      printf ("All stack IDS for %s: ", mpi_function_names[i_mpi]);
      for (int i = 0; i < n_stack_ids[i_mpi]; i++) {
	 printf ("%d ", all_stack_ids_for_i_mpi[i_mpi][i]);
      }
      printf ("\n");
   }
      

   double *t_avg[N_MPI_FUNCS];
   double *max_diff[N_MPI_FUNCS];
   for (int i_mpi = 0; i_mpi < N_MPI_FUNCS; i_mpi++) {
      t_avg[i_mpi] = (double*) malloc (n_stacks[i_mpi][0] * sizeof(double));
      max_diff[i_mpi] = (double*) malloc (n_stacks[i_mpi][0] * sizeof(double));
      for (int i_stack = 0; i_stack < n_stacks[i_mpi][0]; i_stack++) {
         t_avg[i_mpi][i_stack] = 0.0;
	 max_diff[i_mpi][i_stack] = 0.0;
      }
      for (int i_stack = 0; i_stack < n_stacks[i_mpi][0]; i_stack++) {
	 int n = 0;
         for (int i_file = 0; i_file < n_log_files; i_file++) {
	    if (all_t[i_mpi][i_file][i_stack] > 0.0) {
               t_avg[i_mpi][i_stack] += all_t[i_mpi][i_file][i_stack];
	       n++;
            }
         }
	 if (n > 0) t_avg[i_mpi][i_stack] /= n;
	 for (int i_file = 0; i_file < n_log_files; i_file++) {
            if (i_file == 63 && i_mpi == 0 && i_stack == 3) printf ("t: %lf\n", all_t[i_mpi][i_file][i_stack]);
	    if (all_t[i_mpi][i_file][i_stack] > 0.0) {
	       double d = fabs (all_t[i_mpi][i_file][i_stack] - t_avg[i_mpi][i_stack]);
	       if (d > max_diff[i_mpi][i_stack]) {
	         //printf ("max_diff for %lf\n", all_t[i_mpi][i_file][i_stack]);
		 max_diff[i_mpi][i_stack] = d;
	       }
            }
	 }
         bool all_okay = true;
	 for (int i_file = 0; i_file < n_log_files; i_file++) {
	    if (i_mpi != 0) continue;
	    //printf ("%lf ", all_t[i_mpi][i_file][3]);
	    if (all_t[i_mpi][i_file][i_stack] == 0.0) continue;
	//    printf ("%s(%d,%d): %lf %lf %lf %lf %lf\n", mpi_function_names[i_mpi], i_file, i_stack, all_t[i_mpi][i_file][i_stack],
	//					    t_avg[i_mpi][i_stack], max_diff[i_mpi][i_stack], max_diff[i_mpi][i_stack] / t_avg[i_mpi][i_stack] * 100, 
	//				 	    all_imba[i_mpi][i_file][i_stack]);
	//   printf ("%s(%d,%d): %lf %d %lf %d\n", mpi_function_names[i_mpi], i_file, i_stack,
	//	   all_t[i_mpi][i_file][i_stack], all_n_calls[i_mpi][i_file][i_stack],
	//	   all_imba[i_mpi][i_file][i_stack], 0);
	    all_okay = all_okay && equal_for_n_digits (all_imba[i_mpi][i_file][i_stack], max_diff[i_mpi][i_stack] / t_avg[i_mpi][i_stack] * 100, 2);
	    //printf ("%s(%d,%d): OKAY\n", mpi_function_names[i_mpi], i_file, i_stack);
	    }
         if (i_mpi == 0) printf ("\n");
	 if (i_mpi == 0) printf ("%s(%d): %s\n", mpi_function_names[i_mpi], i_stack, all_okay ? "OKAY" : "NOT OKAY");
	 }
      }

   

   return 0;
}
