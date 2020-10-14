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

#define N_MPI_FUNCS 13
char *mpi_function_names[N_MPI_FUNCS] = {"mpi_barrier", "mpi_bcast", "mpi_reduce",
             		                 "mpi_allreduce", "mpi_gatherv", "mpi_gather",
             		                 "mpi_allgatherv", "mpi_allgather",
             		                 "mpi_scatterv", "mpi_scatter",
             		                 "mpi_alltoallv", "mpi_alltoallw", "mpi_alltoall"};

// Be careful: If you rearrange the order of the function names,
// for functions where there is a corresponding one with a suffix
// (like "mpi_scatter" and "mpi_scatterv"), the non-suffixed one
// must come last, because strstr would also be valid with the latter
// one, yielding a wrong index.

int mpi_index (char *s) {
	for (int i = 0; i < N_MPI_FUNCS; i++) {
		if (strstr (s, mpi_function_names[i])) return i;
	}
	return -1;
}	

int main (int argc, char *argv[]) {
	
	int n_log_files = argc - 1;
        double mpi_time[N_MPI_FUNCS][n_log_files];
	double mpi_percentage[N_MPI_FUNCS][n_log_files];
	int n_calls[N_MPI_FUNCS][n_log_files];
	double t_avg[N_MPI_FUNCS][n_log_files];
	double t_min[N_MPI_FUNCS][n_log_files];
	double t_max[N_MPI_FUNCS][n_log_files];
	double imbalance[N_MPI_FUNCS][n_log_files];
	double this_t[N_MPI_FUNCS][n_log_files];

	for (int i = 0; i < N_MPI_FUNCS; i++) {
		for (int j = 0; j < n_log_files; j++) {
		        mpi_percentage[i][j] = 0.0;
			n_calls[i][j] = 0;	
			mpi_time[i][j] = 0.0;
			t_avg[i][j] = 0.0;
			t_min[i][j] = 0.0;
			t_max[i][j] = 0.0;
			imbalance[i][j] = 0.0;
			this_t[i][j] = 0.0;
		}
	}

	for (int i_file = 1; i_file < argc; i_file++) {
		//printf ("Open: %s\n", argv[i_file]);
		FILE *fp = fopen (argv[i_file], "r");
		if (!fp) printf ("SOMETHING WENT WRONG\n");
		char line[256];
		int countdown = -1;
		bool all_mpi_functions_read = false;
		while (!feof(fp)) {
		   //fscanf (fp, "%[^\n]", line);
		   fgets (line, 256, fp);
		   //printf ("This line: %s\n", line);
		   if (countdown < 0) {
			if (strstr (line, "Total time")) {
			   countdown = 4;
			}
		   } else if (countdown > 0) {
			countdown--;
		   } else {
		  	char *column = strtok (line, "|");
			int index = mpi_index (column);
			if (index < 0) {	
				all_mpi_functions_read = true;
			} else {
				column = strtok (NULL, "|");	
				mpi_percentage[index][i_file - 1] = atof(column);
				column = strtok (NULL, "|");
				n_calls[index][i_file - 1] = atoi(column);
				column = strtok (NULL, "|");
				t_avg[index][i_file - 1] = atof(column);
				column = strtok (NULL, "|");
				t_min[index][i_file - 1] = atof(column);
				column = strtok (NULL, "|");
				t_max[index][i_file - 1] = atof(column);
				column = strtok (NULL, "|");
				imbalance[index][i_file - 1] = atof(column);
				column = strtok (NULL, "|");
				this_t[index][i_file - 1] = atof(column);
			}
		   }	
		   if (all_mpi_functions_read) {
			countdown = -1;
			all_mpi_functions_read = false;
			break;
		   }
		}
		fclose (fp);	
	}

	printf ("Check if MPI percentages add up to 100%%:\n");
	bool all_okay = true;
	for (int i_file = 0; i_file < n_log_files; i_file++) {
	   double p_tot = 0.0;
	   for (int i = 0; i < N_MPI_FUNCS; i++) {
	   	p_tot += mpi_percentage[i][i_file];
		//printf ("Add: %lf %lf\n", mpi_percentage[i], p_tot);
	   }
	   // +- 0.05% are tolerable
	   if (p_tot > 100.05 || p_tot < 99.95) {
		all_okay = false;
		printf ("Not okay: %s (%lf)\n", argv[i_file + 1], p_tot);
	   }
	}
 	if (all_okay) printf ("All okay\n");
	//printf ("Check that average times match across all ranks:\n");
	return 0;
}
