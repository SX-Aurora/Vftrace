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

// This program reads in a given number of Vftrace log files which
// belong to the same run. It parses the MPI summary table as well
// as the function profile to perform cross-checks across individual
// functions and ranks. For example, it checks that the individual times
// listed in the summary table are equal to the sum of (inclusive) times
// for that function in the performance overview, or whether the printed
// average time matches the average time computed from the local individual
// times

// These are the functions which can appear in the MPI summary table. They
// are identical to those defined in src/vftr_filewrite.c.
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

/**********************************************************************/

bool equal_within_tolerance (double val1, double val2, double tolerance) {
	return fabs (val1 / val2 - 1.0) < tolerance;
}

/**********************************************************************/

bool equal_for_n_digits (double val1, double val2, int n_digits) {
	double max_diff = 1.0;
	for (int i = 0; i < n_digits; i++) {
	   max_diff /= 10.0;
	}
	return (fabs(val1 - val2) <= max_diff);
}

/**********************************************************************/

bool check_if_mpi_times_add_up (int n_log_files, double mpi_percentage[N_MPI_FUNCS][n_log_files],
				char *filenames[n_log_files + 1]) {
	bool all_okay = true;
	for (int i_file = 0; i_file < n_log_files; i_file++) {
	   double p_tot = 0.0;
	   for (int i = 0; i < N_MPI_FUNCS; i++) {
	   	p_tot += mpi_percentage[i][i_file];
	   }
	   // +- 0.05% are tolerable (Not using equal_within_tolerance because that checks relative deviations)
	   if (p_tot > 100.05 || p_tot < 99.95) {
		all_okay = false;
		printf ("Not okay: %s (%lf)\n", filenames[i_file + 1], p_tot);
	   }
	}
	return all_okay;
}

/**********************************************************************/

// Check average, mininal and maximal time, which should be equal
// for all ranks which participate in a collective call.
	
// We are not (yet) interested in which of the values might be the correct one,
// we only check if they are all equal within a tolerance of 1%.

bool check_if_global_times_match (int n_log_files, double t[N_MPI_FUNCS][n_log_files]) {
	bool all_okay = true;
	for (int i = 0; i < N_MPI_FUNCS; i++) {	
		// Local flag which checks that values for one MPI function match
		bool all_okay_local = true;
		double this_t = t[i][0];
		for (int i_file = 1; i_file < n_log_files && all_okay_local; i_file++) {
			// t == 0 indicates that this rank did not particiticapte in this
			// MPI collective. Therefore, it is skipped.
			if (t[i][i_file] == 0.0) continue;
			// A deviation of one percent is tolerated.
			all_okay_local = equal_within_tolerance (t[i][i_file], this_t, 0.01);
		}
		if (!all_okay_local) printf ("Not okay: %s\n", mpi_function_names[i]);
		all_okay &= all_okay_local;
	}
	return all_okay;
}

/**********************************************************************/

bool check_t_avg (int n_log_files, double t[N_MPI_FUNCS][n_log_files], double t_avg[N_MPI_FUNCS][n_log_files]) {
	bool all_okay = true;	
	for (int i = 0; i < N_MPI_FUNCS; i++) {
		bool all_okay_local = true;
		double sum_t = 0.0;
		int n = 0;
		for (int i_file = 0; i_file < n_log_files; i_file++) {
			if (t[i][i_file] > 0.0) {
				sum_t += t[i][i_file];
				n++;
			}
		}
		if (n > 0) {
			// We accept two percent of deviation, since in Vftrace, the average is
			// computed using the integer (long long) number of microseconds,
			// multiplied by 1e-6, whereas here, we already have floating point
			// numbers.
			double this_t_avg = sum_t / n;
			all_okay_local = equal_within_tolerance (this_t_avg, t_avg[i][0], 0.02);
			if (!all_okay_local) {
		    	   printf ("Not okay: %s %lf %lf\n", mpi_function_names[i],
				   this_t_avg, this_t_avg / t_avg[i][0]);
			}
		}
		all_okay &= all_okay_local;
	}
	return all_okay;
}
		
/**********************************************************************/

bool check_t_min (int n_log_files, double t[N_MPI_FUNCS][n_log_files], double t_min[N_MPI_FUNCS][n_log_files]) {
	bool all_okay = true;	
	for (int i = 0; i < N_MPI_FUNCS; i++) {
		bool all_okay_local = true;
		double this_t_min = DBL_MAX;
		int n = 0;
		int i0 = 0;
		// MPI_Barrier is a bit cumbersome: Often, one rank has a very
		// small registered time, so that the three digits are not
		// sufficient to display it and it appears as zero. 
		// For this reason, we skip MPI_Barrier. If there might be issues
		// with it, the user will have to check it separately.

		if (!strcmp ("mpi_barrier", mpi_function_names[i])) continue;
		while (t_min[i][i0] == 0.0) i0++;
		for (int i_file = 0; i_file < n_log_files; i_file++) {
			if (t[i][i_file] > 0.0 && t[i][i_file] < this_t_min) {
				this_t_min = t[i][i_file]; 
				n++;
			}
		}
		if (n > 0) {
			all_okay_local = equal_within_tolerance (this_t_min, t_min[i][i0], 0.02);
			if (!all_okay_local) {
			   printf ("Not okay: %s(%d) %lf %lf %lf\n", mpi_function_names[i],
			   i, this_t_min, t_min[i][i0], this_t_min / t_min[i][i0]);
			}
		}
		all_okay &= all_okay_local;
	}
	return all_okay;
}
		
/**********************************************************************/

bool check_t_max (int n_log_files, double t[N_MPI_FUNCS][n_log_files], double t_max[N_MPI_FUNCS][n_log_files]) {
	bool all_okay = true;	
	for (int i = 0; i < N_MPI_FUNCS; i++) {
		bool all_okay_local = true;
		double this_t_max = 0.0;
		int n = 0;
		int i0 = 0;
		while (t_max[i][i0] == 0.0) i0++;
		for (int i_file = 0; i_file < n_log_files; i_file++) {
			if (t[i][i_file] > this_t_max) {
				this_t_max = t[i][i_file]; 
				n++;
			}
		}
		if (n > 0) {
			all_okay_local = equal_within_tolerance (this_t_max, t_max[i][i0], 0.02);
			if (!all_okay_local) printf ("Not okay: %s(%d) %lf %lf %lf\n", 
			    mpi_function_names[i], i, this_t_max, t_max[i][i0], this_t_max / t_max[i][i0]);
		}
		all_okay &= all_okay_local;
	}
	return all_okay;
}
		
/**********************************************************************/
#define LINEBUFSIZE 256

void check_each_time (FILE *fp, int n_calls_vftr[N_MPI_FUNCS], double t_tot[N_MPI_FUNCS], double t_tot_all,
		      bool *all_calls_okay, bool *all_t_okay, bool *t_tot_okay) {
	char line[LINEBUFSIZE];
	int countdown = -1;	
	int n_calls, n_calls_tot[N_MPI_FUNCS];
	double t_inc, t_inc_tot[N_MPI_FUNCS];
	double this_t_tot = 0.0;

 	for (int i = 0; i < N_MPI_FUNCS; i++) {
	   n_calls_tot[i] = 0;
	   t_inc_tot[i] = 0.0;
        }

	while (!feof(fp)) {
  	   fgets (line, LINEBUFSIZE, fp);
	   if (countdown < 0) {
		// When this string is found, there are five more lines afterwards
		// before the actual profile starts.
		if (strstr (line, "Runtime profile for rank")) {
			countdown = 4;
	        }
   	   } else if (countdown > 0) {
	      countdown--;
	   } else {
	      if (strstr(line, "----")) break;
	      char *column;
	      column = strtok (line, " "); // #Calls
	      n_calls = atoi (column);	
	      column = strtok (NULL, " "); // Exclusive time, skip
	      column = strtok (NULL, " "); // Inclusive time
	      t_inc = atof (column);
	      column = strtok (NULL, " "); // %abs, skip
	      column = strtok (NULL, " "); // %rel, skip
	      column = strtok (NULL, " "); // Function name
	      int index;
	      if ((index = mpi_index(column)) >= 0) {
		 n_calls_tot[index] += n_calls;
		 t_inc_tot[index] += t_inc;
	      }
	   }
        }

	*all_calls_okay = true;
	*all_t_okay = true;
	for (int i = 0; i < N_MPI_FUNCS; i++) {
	   *all_calls_okay &= (n_calls_tot[i] == n_calls_vftr[i]); 
	   bool tmp = equal_for_n_digits (t_inc_tot[i], t_tot[i], 2);
	   if (!tmp) {
		printf ("Added and registered time are not equal(%d): %lf %lf\n",
			i, t_inc_tot[i], t_tot[i]);
	   }
	   *all_t_okay &= tmp;
	   this_t_tot += t_inc_tot[i];
	}
	*t_tot_okay = equal_for_n_digits (this_t_tot, t_tot_all, 1);
	if (!*t_tot_okay) {
		printf ("Total MPI does not match: %lf %lf\n", this_t_tot, t_tot_all);
	}
}

/**********************************************************************/

bool check_if_imbalances_match (int n_log_files, double t[N_MPI_FUNCS][n_log_files],
			        double t_avg[N_MPI_FUNCS][n_log_files],
				double imbalances[N_MPI_FUNCS][n_log_files]) {
	double max_diff[N_MPI_FUNCS];
	bool all_okay = true;
	for (int i = 0; i < N_MPI_FUNCS; i++) {
		// As above (check_t_min), we skip MPI_Barrier.
		if (!strcmp ("mpi_barrier", mpi_function_names[i])) continue;
		bool all_okay_local = true;
		max_diff[i] = 0.0;
		int i0 = 0;
		while (t_avg[i][i0] == 0.0) i0++;
		for (int i_file = 0; i_file < n_log_files; i_file++) {
			if (t[i][i_file] > 0.0) {
				double d = fabs(t[i][i_file] - t_avg[i][i0]);
				if (d > max_diff[i]) max_diff[i] = d;	
			}	
		}
		double this_imba = max_diff[i] / t_avg[i][i0] * 100.0;
		for (int i_file = 0; i_file < n_log_files; i_file++) {
			if (t[i][i_file] == 0.0) continue; // If the function does not occur in this rank
			bool tmp = equal_within_tolerance (this_imba, imbalances[i][i_file], 0.05);
			if (!tmp) printf ("Not okay: %s(%d), %d, Time: %lf, This imba: %lf, Registered imba: %lf\n",
					  mpi_function_names[i], i, i_file, t[i][i_file], this_imba, imbalances[i][i_file]);
			all_okay_local &= tmp;
		}
		all_okay &= all_okay_local;
	}	
	return all_okay;
}
/**********************************************************************/

int main (int argc, char *argv[]) {
	
	// Read in all these fields from the MPI table in the given log files.
	int n_log_files = argc - 1;
        double mpi_time[N_MPI_FUNCS][n_log_files];
	double mpi_percentage[N_MPI_FUNCS][n_log_files];
	int n_calls[N_MPI_FUNCS][n_log_files];
	double t_avg[N_MPI_FUNCS][n_log_files];
	double t_min[N_MPI_FUNCS][n_log_files];
	double t_max[N_MPI_FUNCS][n_log_files];
	double imbalance[N_MPI_FUNCS][n_log_files];
	double this_t[N_MPI_FUNCS][n_log_files];
   	double t_tot[n_log_files];

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
			t_tot[j] = 0.0;
		}
	}

	bool prof_truncated = false;
	for (int i_file = 1; i_file < argc; i_file++) {
		FILE *fp = fopen (argv[i_file], "r");
		if (!fp) {
			printf ("Error: Could not open %s\n", argv[i_file]);
			return -1;
		}
		char line[LINEBUFSIZE];
		int countdown = -1;
		char *column;
		bool all_mpi_functions_read = false;
		while (!feof(fp)) {
		   fgets (line, LINEBUFSIZE, fp);
		   if (countdown < 0) {
			// When this string is found, there are four more lines afterwards
			// before the actual profile starts.
			if (strstr (line, "Total time")) {
			   countdown = 4;
			   column = strtok (line, " ");   
			   int i = 0;	
			   while (i++ < 8) column = strtok (NULL, " ");
			   t_tot[i_file - 1] = atof (column);
			} else if (!prof_truncated && strstr (line, "Runtime profile")) {
			   prof_truncated = strstr(line, "truncated");
			}
		   } else if (countdown > 0) {
			countdown--;
		   } else {
		  	column = strtok (line, "|");
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

	// All data has been read in. Now begin with the consistency checks.

	bool all_okay;
	printf ("Check if MPI percentages add up to 100%%:\n");
	all_okay = check_if_mpi_times_add_up (n_log_files, mpi_percentage, argv);
 	if (all_okay) printf ("All okay\n");

	printf ("Check that average times match across all ranks:\n");
	all_okay = check_if_global_times_match (n_log_files, t_avg);
	if (all_okay) {
		printf ("All okay\n");
		printf ("Check if average time is reproudced: %s\n",
			 check_t_avg (n_log_files, this_t, t_avg) ? "YES" : "NO");
	}
	
	printf ("Check that minimum times match across all ranks:\n");
	all_okay = check_if_global_times_match (n_log_files, t_min);
	if (all_okay) {
		printf ("All okay\n");
		printf ("Check if minimum time is reproudced: %s\n",
			 check_t_min (n_log_files, this_t, t_min) ? "YES" : "NO");
	}
	
	printf ("Check that maximum times match across all ranks:\n");
	all_okay = check_if_global_times_match (n_log_files, t_max);
	if (all_okay) {
		printf ("All okay\n");
		printf ("Check if maximum time is reproudced: %s\n",
			 check_t_max (n_log_files, this_t, t_max) ? "YES" : "NO");
	}

	if (prof_truncated) {
	   printf ("The runtime profile is truncated, which means that not all functions are listed.\n");
	   printf ("The intra-file crosschecks will be inaccurate and are skipped.\n");
	   printf ("Redo the Vftrace measurement with VFTR_PROF_TRUNCATE=no in order to execute them.\n");
	   return 0;
	}

	for (int i_file = 0; i_file < n_log_files; i_file++) {
	   printf ("Check that time values add up to the registered values: %s\n", argv[i_file + 1]);
	   FILE *fp = fopen (argv[i_file + 1], "r");
	   int n_calls_this_rank[N_MPI_FUNCS];
	   double this_t_this_rank[N_MPI_FUNCS];
	   for (int i = 0; i < N_MPI_FUNCS; i++) {
	   	n_calls_this_rank[i] = n_calls[i][i_file];
	   	this_t_this_rank[i] = this_t[i][i_file];
	   }
	   bool all_calls_okay, all_t_okay, t_tot_okay;	
	   check_each_time (fp, n_calls_this_rank, this_t_this_rank, t_tot[i_file],  
	   		    &all_calls_okay, &all_t_okay, &t_tot_okay);
	   printf ("Calls okay: %s\n", all_calls_okay ? "YES" : "NO");
	   printf ("Times okay: %s\n", all_t_okay ? "YES" : "NO");
	   printf ("Total time okay: %s\n", t_tot_okay ? "YES" : "NO");
	   fclose (fp);
        }

	printf ("Check imbalances: \n");
	all_okay = check_if_imbalances_match (n_log_files, this_t, t_avg, imbalance);
	if (all_okay) printf ("All okay\n");
	
	return 0;
}
