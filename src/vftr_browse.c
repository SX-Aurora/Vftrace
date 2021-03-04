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

#include <string.h>
#include <math.h>
#include <sys/stat.h>

#include "vftr_environment.h"
#include "vftr_hwcounters.h"
#include "vftr_scenarios.h"
#include "vftr_browse.h"
#include "vftr_setup.h"
#include "vftr_functions.h"
#include "vftr_filewrite.h"
#include "vftr_fileutils.h"
#include "vftr_stacks.h"

/**********************************************************************/

#define TREE_EXTRA_SPACES 10
#define CSS_DEFAULT_WIDTH 1000

void vftr_browse_print_css_header (FILE *fp, int n_chars_max, int n_final) {

   fprintf (fp, "<style>\n");
   fprintf (fp, "*,\n");
   fprintf (fp, "*:before,\n");
   fprintf (fp, "*:after {\n");
   fprintf (fp, "  -webkit-box-sizing: border-box;\n");
   fprintf (fp, "  -moz-box-sizing: border-box;\n");
   fprintf (fp, "  box-sizing: border-box;\n");
   fprintf (fp, "}\n");
   fprintf (fp, "\n");
   fprintf (fp, "* {\n");
   fprintf (fp, "  position: relative;\n");
   fprintf (fp, "  margin: 0;\n");
   fprintf (fp, "  padding: 0;\n");
   fprintf (fp, "\n"); 
   fprintf (fp, "  border: 0 none;\n");
   fprintf (fp, "\n"); 
   fprintf (fp, "  -webkit-transition: all ease .4s;\n");
   fprintf (fp, "  -moz-transition: all ease .4s;\n");
   fprintf (fp, "    transition: all ease .4s;\n");
   fprintf (fp, "}\n");
   fprintf (fp, "\n");
   fprintf (fp, "html, body {\n");
   fprintf (fp, "  width: 100%%;\n");
   fprintf (fp, "  height: 100%%;\n");
   fprintf (fp, "  background: white;\n");
   fprintf (fp, "  font-family: 'Fjalla One', sans-serif;\n");
   fprintf (fp, "  font-size: 18px;\n");
   fprintf (fp, "}\n");
   fprintf (fp, "\n");
   fprintf (fp, "h1 {\n");
   fprintf (fp, "  padding-top: 40px;\n");
   fprintf (fp, "  color: black;\n");
   fprintf (fp, "  text-align: center;\n");
   fprintf (fp, "  font-size: 1.8rem;\n");
   fprintf (fp, "}\n");
   fprintf (fp, "\n");
   fprintf (fp, ".nav {\n");
   fprintf (fp, "  margin: 100px auto;\n");
   fprintf (fp, "  width: %dch;\n", (n_chars_max + TREE_EXTRA_SPACES) * n_final);
   fprintf (fp, "  min-height: auto;\n");
   fprintf (fp, "}\n");
   fprintf (fp, "\n");
   fprintf (fp, ".nav ul {\n");
   fprintf (fp, "  position: relative;\n");
   fprintf (fp, "  padding-top: 20px;\n");
   fprintf (fp, "}\n");
   fprintf (fp, "\n");
   fprintf (fp, ".nav li {\n");
   fprintf (fp, "  position: relative;\n");
   fprintf (fp, "  padding: 20px 3px 0 3px;\n");
   fprintf (fp, "  float: left;\n");
   fprintf (fp, "\n");
   fprintf (fp, "  text-align: center;\n");
   fprintf (fp, "  list-style-type: none;\n");
   fprintf (fp, "}\n");
   fprintf (fp, "\n");
   fprintf (fp, ".nav li::before, .nav li::after{\n");
   fprintf (fp, "  content: '';\n");
   fprintf (fp, "  position: absolute;\n");
   fprintf (fp, "  top: 0;\n");
   fprintf (fp, "  right: 50%%;\n");
   fprintf (fp, "  width: 50%%;\n");
   fprintf (fp, "  height: 20px;\n");
   fprintf (fp, "  border-top: 1px solid black;\n");
   fprintf (fp, "}\n");
   fprintf (fp, "\n");
   fprintf (fp, ".nav li::after{\n");
   fprintf (fp, "  left: 50%%;\n");
   fprintf (fp, "  right: auto;\n");
   fprintf (fp, "\n");  
   fprintf (fp, "  border-left: 1px solid black;\n");
   fprintf (fp, "}\n");
   fprintf (fp, "\n");
   fprintf (fp, ".nav li:only-child::after, .nav li:only-child::before {\n");
   fprintf (fp, "  content: '';\n");
   fprintf (fp, "  display: none;\n");
   fprintf (fp, "}\n");
   fprintf (fp, "\n");
   fprintf (fp, ".nav li:only-child{ padding-top: 0;}\n");
   fprintf (fp, ".nav li:first-child::before, .nav li:last-child::after{\n");
   fprintf (fp, "  border: 0 none;\n");
   fprintf (fp, "}\n");
   fprintf (fp, "\n");
   fprintf (fp, ".nav li:last-child::before{\n");
   fprintf (fp, "  border-right: 1px solid black;\n");
   fprintf (fp, "  border-radius: 0 5px 0 0;\n");
   fprintf (fp, "}\n");
   fprintf (fp, "\n");
   fprintf (fp, ".nav li:first-child::after{\n");
   fprintf (fp, "    border-radius: 5px 0 0 0;\n");
   fprintf (fp, "}\n");
   fprintf (fp, ".nav ul ul::before{\n");
   fprintf (fp, "  content: '';\n");
   fprintf (fp, "  position: absolute; top: 0; left: 50%%;\n");
   fprintf (fp, "  border-left: 1px solid black;\n");
   fprintf (fp, "  width: 0;\n");
   fprintf (fp, "  height: 20px;\n");
   fprintf (fp, "}\n");
   fprintf (fp, "\n");
   fprintf (fp, ".nav li a{\n");
   fprintf (fp, "  display: inline-block;\n");
   fprintf (fp, "  padding: 5px 10px;\n");
   fprintf (fp, "  border-radius: 5px;\n");
   fprintf (fp, "  border: 1px solid black;\n");
   fprintf (fp, "  text-decoration: none;\n");
   fprintf (fp, "  text-transform: uppercase;\n");
   fprintf (fp, "  color: black;\n");
   fprintf (fp, "  font-family: arial, verdana, tahoma;\n");
   fprintf (fp, "  font-size: 11px;\n");
   fprintf (fp, "}\n");
   fprintf (fp, "\n");
   fprintf (fp, "/* This is the stack info box which appears when hovering over a function node */\n");
   fprintf (fp, "\n");
   fprintf (fp, "a .ttt {\n");
   fprintf (fp, "  visibility: hidden;\n");
   fprintf (fp, "  width: 100%%;\n");
   fprintf (fp, "  background-color: black;\n");
   fprintf (fp, "  color: #fff;\n");
   fprintf (fp, "  text-align: center;\n");
   fprintf (fp, "  border-radius: 6px; /* Rounded corners */\n");
   fprintf (fp, "  padding: 10px 0;\n");
   fprintf (fp, "  position: absolute;\n");
   fprintf (fp, "  z-index: 1; /* Precedence of this element - atop of the leaf. */\n");
   fprintf (fp, "}\n");
   fprintf (fp, "\n");
   fprintf (fp, "a:hover .ttt{\n");
   fprintf (fp, "  visibility: visible;\n");
   fprintf (fp, "}\n");
   fprintf (fp, "\n");
   fprintf (fp, "hr {\n");
   fprintf (fp, "  display: block;\n");
   fprintf (fp, "  height: 1px;\n");
   fprintf (fp, "  border: 0;\n");
   fprintf (fp, "  border-top: 1px solid black;\n");
   fprintf (fp, "  margin: 1em 0;\n");
   fprintf (fp, "  padding: 0;\n");
   fprintf (fp, "}\n");

   fprintf (fp, "\n");
   fprintf (fp, "li a, .dropbtn {\n");
   fprintf (fp, "  display: inline-block;\n");
   fprintf (fp, "  color: gray;\n");
   fprintf (fp, "  text-align: center;\n");
   fprintf (fp, "  padding: 14px 16px;\n");
   fprintf (fp, "  text-decoration: none;\n");
   fprintf (fp, "}\n");
   fprintf (fp, "\n");
   fprintf (fp, ".dropdown-content {\n");
   fprintf (fp, "  display: none;\n");
   fprintf (fp, "  position: absolute;\n");
   fprintf (fp, "  background-color: #f9f9f9;\n");
   fprintf (fp, "  min-width: 160px;\n");
   fprintf (fp, "  z-index: 1;\n");
   fprintf (fp, "}\n");
   fprintf (fp, "\n");
   fprintf (fp, ".dropdown-content a:hover {\n");
   fprintf (fp, "  background-color: #f1f1f1;\n");
   fprintf (fp, "}\n");
   fprintf (fp, "\n");
   fprintf (fp, ".dropdown:hover .dropdown-content {\n");
   fprintf (fp, "  display: block;\n");
   fprintf (fp, "}\n");
   fprintf (fp, "\n");
   fprintf (fp, "li.dropdown {\n");
   fprintf (fp, "  display: inline-block;\n");
   fprintf (fp, "}\n");

   fprintf (fp, "</style>\n");
}

/**********************************************************************/

#define INDENT_SPACES 3

void vftr_browse_make_html_indent (FILE *fp, int n_indent_0, int n_indent_extra) {
	for (int i = 0; i < n_indent_0 + n_indent_extra * INDENT_SPACES; i++) fprintf (fp, " ");
}

/**********************************************************************/


void vftr_create_callgraph_dropdown (FILE *fp, char *func_name, enum origin_page op) {
   if (op == HOME || op == PROFILE) { 
      vftr_browse_make_html_indent (fp, 0, 1);
      fprintf (fp, "<li style=\"display: inline;\">\n"); 
      vftr_browse_make_html_indent (fp, 0, 2);
      int n = strlen(func_name) + 8;
      char target_fun[n];
      snprintf (target_fun, n, "%s_%d.html", func_name, 0);
      fprintf (fp, "<a href=\"%s/%s\">Call graphs</a>\n", func_name, target_fun);
      vftr_browse_make_html_indent (fp, 0, 2);
      fprintf (fp, "</li>\n");
   } else {
      vftr_browse_make_html_indent (fp, 0, 1);
      fprintf (fp, "<li style=\"display: inline;\" class=\"dropdown\">\n"); 
      vftr_browse_make_html_indent (fp, 0, 2);
      fprintf (fp, "<a href=\"javascript:void(0)\" class=\"dropbtn\">Call graphs</a>\n");
      vftr_browse_make_html_indent (fp, 0, 2);
      fprintf (fp, "<div class=\"dropdown-content\">\n");
      for (int i = 0; i < vftr_mpisize; i++) {
         vftr_browse_make_html_indent (fp, 0, 3);
         int n = strlen(func_name) + vftr_count_digits_int(i) + 7;
         char target_fun[n];
         snprintf (target_fun, n, "%s_%d.html", func_name, i);
         fprintf (fp, "<a href=\"%s\">%d</a>\n", target_fun, i);
      }
      vftr_browse_make_html_indent (fp, 0, 2);
      fprintf (fp, "</div>\n");
      vftr_browse_make_html_indent (fp, 0, 1);
      fprintf (fp, "</li>\n");
   }
}

/**********************************************************************/

void vftr_create_profile_dropdown (FILE *fp, enum origin_page op) {
    if (op == HOME) {
      vftr_browse_make_html_indent (fp, 0, 1);
      fprintf (fp, "<li style=\"display: inline;\">\n");
      vftr_browse_make_html_indent (fp, 0, 2);
      fprintf (fp, "<a href=\"profile_%d.html\">Profile tables</a>\n", vftr_mpirank);
      vftr_browse_make_html_indent (fp, 0, 1);
      fprintf (fp, "</li>\n");
    } else if (op == TREE) {
      vftr_browse_make_html_indent (fp, 0, 1);
      fprintf (fp, "<li style=\"display: inline;\">\n");
      vftr_browse_make_html_indent (fp, 0, 2);
      fprintf (fp, "<a href=\"../profile_%d.html\">Profile tables</a>\n", vftr_mpirank);
      vftr_browse_make_html_indent (fp, 0, 1);
      fprintf (fp, "</li>\n");
   } else {
      vftr_browse_make_html_indent (fp, 0, 1);
      fprintf (fp, "<li style=\"display: inline;\" class=\"dropdown\">\n"); 
      vftr_browse_make_html_indent (fp, 0, 2);
      fprintf (fp, "<a href=\"javascript:void(0)\" class=\"dropbtn\">Profile tables</a>\n");
      vftr_browse_make_html_indent (fp, 0, 2);
      fprintf (fp, "<div class=\"dropdown-content\">\n");
      for (int i = 0; i < vftr_mpisize; i++) {
         vftr_browse_make_html_indent (fp, 0, 3);
         fprintf (fp, "<a href=\"profile_%d.html\">%d</a>\n", i, i);
      }
      vftr_browse_make_html_indent (fp, 0, 2);
      fprintf (fp, "</div>\n");
      vftr_browse_make_html_indent (fp, 0, 1);
      fprintf (fp, "</li>\n");
   }
}

/**********************************************************************/

void vftr_browse_print_navigation_bars (FILE *fp, display_function_t **display_functions, int i_func, int n_funcs, enum origin_page op) {
   // Horizontal navigation bar
   fprintf (fp, "<ul style=\"list-style-type: none;margin-top: 0px;margin-left: 150px; background-color: #f1f1f1;\">\n");
   vftr_browse_make_html_indent (fp, 0, 1);
   if (op == HOME || op == PROFILE) {
      fprintf (fp, "<li style=\"display: inline;\"><a href=\"index.html\">HOME</a></li>\n");
   } else {
      fprintf (fp, "<li style=\"display: inline;\"><a href=\"../index.html\">HOME</a></li>\n");
   }
   vftr_create_callgraph_dropdown (fp, display_functions[i_func]->func_name, op);
   vftr_create_profile_dropdown (fp, op);
   fprintf (fp, "</ul>\n");
   fprintf (fp, "\n");
   // Vertical navigation bar
   fprintf (fp, "<ul style=\"float: left;list-style-type: none;margin: 0;padding: 0; width: 150px;background-color: #f1f1f1;\">\n");
   for (int i = 0; i < n_funcs; i++) {
      char *func_name = display_functions[i]->func_name;
      vftr_browse_make_html_indent (fp, 0, 1);
      int n = strlen (func_name) + vftr_count_digits_int(vftr_mpirank) + 7;
      char target_fun[n];
      snprintf (target_fun, n, "%s_%d.html", func_name, vftr_mpirank);
      if (op == HOME) {
         fprintf (fp, "<li><a href=\"%s/%s\">%s</a></li>\n", func_name, target_fun, func_name);
      } else if (i == i_func) {
         fprintf (fp, "<li><a href=\"%s\">%s</a></li>\n", target_fun, func_name);
      } else {
         fprintf (fp, "<li><a href=\"../%s/%s\">%s</a></li>\n", func_name, target_fun, func_name);
      }	
   }
   fprintf (fp, "</ul>\n");
   fprintf (fp, "\n");
}

/**********************************************************************/

void vftr_browse_print_index_html (display_function_t **display_functions, int n_funcs) {
   FILE *fp = fopen ("browse/index.html", "w+");
   vftr_browse_print_css_header (fp, CSS_DEFAULT_WIDTH, 1);
   vftr_browse_print_navigation_bars (fp, display_functions, 0, n_funcs, HOME);
   fprintf (fp, "<div stype=\"margin-left:150px; padding 1px, 16px\">\n");
   vftr_browse_make_html_indent (fp, 0, 1);
   fprintf (fp, "<h1>Vfbrowse</h1>\n");
   vftr_browse_make_html_indent (fp, 0, 1);
   fprintf (fp, "<p>Application: %s</p>\n", vftr_get_program_path ());
   vftr_browse_make_html_indent (fp, 0, 1);
   fprintf (fp, "<p>Start date: %s</p>\n", vftr_start_date);
   vftr_browse_make_html_indent (fp, 0, 1);
   fprintf (fp, "<p>End date: %s</p>\n", vftr_end_date);
   vftr_browse_make_html_indent (fp, 0, 1);
   fprintf (fp, "<p>Number of MPI ranks: %d</p>\n", vftr_mpisize);

   long long total_runtime_usec, sampling_overhead_time_usec, total_overhead_time_usec;
   long long mpi_overhead_time_usec, application_runtime_usec;
   vftr_get_application_times_usec (0, &total_runtime_usec, &sampling_overhead_time_usec, &mpi_overhead_time_usec, 
			            &total_overhead_time_usec, &application_runtime_usec);
   vftr_browse_make_html_indent (fp, 0, 1);
   fprintf (fp, "<p>Total runtime: %8.2f seconds</p>\n", total_runtime_usec * 1e-6);
   vftr_browse_make_html_indent (fp, 0, 1);
   fprintf (fp, "<p>Application time: %8.2f seconds</p>\n", application_runtime_usec * 1e-6);
   vftr_browse_make_html_indent (fp, 0, 1);
   fprintf (fp, "<p>Overhead: %8.2f seconds (%.2f%%)</p>\n", total_overhead_time_usec * 1e-6, total_overhead_time_usec / total_runtime_usec * 100.0);
   vftr_browse_make_html_indent (fp, 0, 1);
   fprintf (fp, "<p>Sampling overhead: %8.2f seconds (%.2f%%)</p>\n", sampling_overhead_time_usec * 1e-6,
	    sampling_overhead_time_usec / total_runtime_usec * 100.0);
   vftr_browse_make_html_indent (fp, 0, 1);
   fprintf (fp, "<p>MPI overhead: %8.2f seconds (%.2f%%)</p>\n", mpi_overhead_time_usec * 1e-6,
	   mpi_overhead_time_usec / total_runtime_usec * 100.0);
   fprintf (fp, "</div>\n");
   fclose (fp); 
}

/**********************************************************************/

void vftr_browse_print_tree_element (FILE *fp, int final_id, int stack_id, int func_id, int n_spaces, double total_time) {
	vftr_browse_make_html_indent (fp, n_spaces, 0);
	if (final_id > 0) {
	   double this_t = func_id >= 0 ? vftr_func_table[func_id]->prof_current.time_incl * 1e-6 : 0.0;
	   int x = total_time > 0 ? (int)(floor (510 * this_t / total_time)) : 0;
	   int rvalue = x > 255 ? 255 : 255 - x;
	   int gvalue = x > 255 ? 510 - x : 255;
	   fprintf (fp, "<a hfref=\"#\" style=\"background-color: rgb(%d,%d,0)\">%s\n",
		    rvalue, gvalue, vftr_gStackinfo[stack_id].name);
	} else {
	   fprintf (fp, "<a hfref=\"#\" style=\"background-color: #edebeb\">%s\n", vftr_gStackinfo[stack_id].name);
	}
	//vftr_browse_make_html_indent (fp, n_spaces + 3, 0);
	//fprintf (fp, "<ttt class=\"ttt\">function name: %s\n", vftr_gStackinfo[stack_id].name);
	//vftr_browse_make_html_indent (fp, n_spaces + 3, 0);
	//if (func_id < 0) {
	//   fprintf (fp, "<br>n_calls: %d\n", -1);
	//} else {
	//   fprintf (fp, "<br>n_calls: %lld\n", vftr_func_table[func_id]->prof_current.calls);
	//}
	//vftr_browse_make_html_indent (fp, n_spaces + 3, 0);
	//fprintf (fp, "</ttt>\n");
	vftr_browse_make_html_indent (fp, n_spaces, 0);
	fprintf (fp, "</a>\n");
}

/**********************************************************************/

void vftr_browse_print_stacktree (FILE *fp, stack_leaf_t *leaf, int n_spaces, double total_time, double *imbalances) {
	if (!leaf) return;
	vftr_browse_print_tree_element (fp, leaf->final_id, leaf->stack_id, leaf->func_id, n_spaces, total_time);
	if (leaf->callee) {
		vftr_browse_make_html_indent (fp, n_spaces, 0);
		fprintf (fp, "<ul>\n");
		vftr_browse_make_html_indent (fp, n_spaces, 1);
		fprintf (fp, "<li>\n");
		vftr_browse_print_stacktree (fp, leaf->callee, n_spaces + 2 * INDENT_SPACES, total_time, imbalances);
		vftr_browse_make_html_indent (fp, n_spaces, 1);
		fprintf (fp, "</li>\n");
		vftr_browse_make_html_indent (fp, n_spaces, 0);
		fprintf (fp, "</ul>\n");
	} else {
	       vftr_browse_make_html_indent (fp, n_spaces, 0);
	       fprintf (fp, "<hr/>\n");
	       if (leaf->func_id < 0) { 
	          vftr_browse_make_html_indent (fp, n_spaces, 0);
		  fprintf (fp, "[not on this rank]\n");
	       } else { 
	          vftr_browse_make_html_indent (fp, n_spaces, 0);
	          fprintf (fp, "Position: %d<br>\n", leaf->final_id);
	          vftr_browse_make_html_indent (fp, n_spaces, 0);
		  fprintf (fp, "Calls: %lld<br>\n", vftr_func_table[leaf->func_id]->prof_current.calls);
	          vftr_browse_make_html_indent (fp, n_spaces, 0);
		  double t = vftr_func_table[leaf->func_id]->prof_current.time_incl * 1e-6;
		  char *t_unit;
		  vftr_time_unit (&t, &t_unit, true); 
		  fprintf (fp, "Time %6.2f %s<br>\n", t, t_unit);
	          vftr_browse_make_html_indent (fp, n_spaces, 0);
		  if (imbalances != NULL) {
			if (imbalances[leaf->func_id] > 0.0) {
			   fprintf (fp, "Imbalance: %6.2f %%<br>\n", imbalances[leaf->func_id]);
			} else {
			   fprintf (fp, "Imbalance: -/-<br>\n");
			}
		  }	
		  double n_bytes = vftr_func_table[leaf->func_id]->prof_current.mpi_tot_send_bytes;
		  char *unit_str;
		  vftr_memory_unit (&n_bytes, &unit_str);
	          vftr_browse_make_html_indent (fp, n_spaces, 0);
		  fprintf (fp, "Send: %*.2f %s<br>\n", vftr_count_digits_double (n_bytes), n_bytes, unit_str);
		  n_bytes = vftr_func_table[leaf->func_id]->prof_current.mpi_tot_recv_bytes;
		  vftr_memory_unit (&n_bytes, &unit_str);
	          vftr_browse_make_html_indent (fp, n_spaces, 0);
		  fprintf (fp, "Recv: %*.2f %s<br>\n", vftr_count_digits_double(n_bytes), n_bytes, unit_str);
	     }
	}

	if (leaf->next_in_level) {
		vftr_browse_make_html_indent (fp, n_spaces, 0);
		fprintf (fp, "<li>\n");
		vftr_browse_print_stacktree (fp, leaf->next_in_level, n_spaces + INDENT_SPACES, total_time, imbalances);
		vftr_browse_make_html_indent (fp, n_spaces, 0);
		fprintf (fp, "</li>\n");
	}
}
  
/**********************************************************************/

void vftr_browse_print_stacktree_page (FILE *fp_out, bool is_empty, display_function_t **display_functions, int i_func, int n_funcs,
			               stack_leaf_t *leaf, double *imbalances, double total_time,
				       int n_chars_max, int n_final) {
	// No external file (e.g. for tests) given. Create filename <func_name>.html
	FILE *fp;
	char *func_name = display_functions[i_func]->func_name;
	if (!fp_out) {
	   char outdir[strlen(func_name) + 8];
	   snprintf (outdir, strlen(func_name) + 8, "browse/%s", func_name);
	   if (vftr_mpirank == 0) {
	      mkdir (outdir, 0777);
	   }
#ifdef _MPI
	   PMPI_Barrier(MPI_COMM_WORLD);
#endif
	   char html_filename[2*(strlen(func_name) + 8)];
	   snprintf (html_filename, 2*(strlen(func_name) + 8) + vftr_count_digits_int(vftr_mpisize) + 1,
		     "%s/%s_%d.html", outdir, func_name, vftr_mpirank);
	   fp = fopen (html_filename, "w");
        } else {
	   fp = fp_out;
        }
	if (!is_empty) {
	   vftr_browse_print_css_header (fp, n_chars_max, n_final);
	} else {
	   vftr_browse_print_css_header (fp, strlen ("No stack IDs!"), 1);
	}
        vftr_browse_print_navigation_bars (fp, display_functions, i_func, n_funcs, TREE);
	fprintf (fp, "<div style=\"margin-left:150px; margin-top:0px; padding: 1px 16px\">\n");
	fprintf (fp, "<h1>%s, rank %d</h1>\n", func_name, vftr_mpirank);
	fprintf (fp, "<nav class=\"nav\"/>\n");
	vftr_browse_make_html_indent (fp, 0, 1);
	fprintf (fp, "<ul>\n");
	vftr_browse_make_html_indent (fp, 0, 2);
	fprintf (fp, "<li>\n");
	if (!is_empty) {
	   vftr_browse_print_stacktree (fp, leaf->origin, 3 * INDENT_SPACES, total_time, imbalances);
        } else {
	   fprintf (fp, "<h2>No stack IDs!\n</h2>");
	}
	vftr_browse_make_html_indent (fp, 0, 2);
	fprintf (fp, "</li>\n");
	vftr_browse_make_html_indent (fp, 0, 1);
	fprintf (fp, "</ul>\n");
	fprintf (fp, "</nav>\n");
	fprintf (fp, "</div>\n");
	if (!fp_out) fclose (fp); //External file is supposed to be closed elsewhere
}

/**********************************************************************/

void vftr_browse_create_directory () {
       if (vftr_mpirank == 0) {
	  mkdir ("browse", 0777);
       }
#ifdef _MPI
       PMPI_Barrier (MPI_COMM_WORLD);
#endif
}

/**********************************************************************/

FILE *vftr_browse_init_profile_table (display_function_t **display_functions, int n_display_functions) {
       FILE *fp;
       int n = 21 + vftr_count_digits_int(vftr_mpirank);
       char html_profile[n];
       snprintf (html_profile, n, "browse/profile_%d.html", vftr_mpirank);
       fp = fopen (html_profile, "w+");
       vftr_browse_print_css_header (fp, CSS_DEFAULT_WIDTH, 1);
       fprintf (fp,"<style>\n");
       vftr_browse_make_html_indent (fp, 0, 1);
       fprintf (fp, "table {\n");
       vftr_browse_make_html_indent (fp, 0, 2);
       fprintf (fp, "border-collapse: collapse;\n");
       vftr_browse_make_html_indent (fp, 0, 1);
       fprintf (fp, "}\n");
       vftr_browse_make_html_indent (fp, 0, 1);
       fprintf (fp, "th, td {\n");
       vftr_browse_make_html_indent (fp, 0, 2);
       fprintf (fp, "border: 1px solid black;\n");
       vftr_browse_make_html_indent (fp, 0, 2);
       fprintf (fp, "border-collapse: collapse;\n");
       vftr_browse_make_html_indent (fp, 0, 1);
       fprintf (fp, "padding: 2px 12px 2px 23px;\n");
       vftr_browse_make_html_indent (fp, 0, 1);
       fprintf (fp, "}\n");
       vftr_browse_make_html_indent (fp, 0, 1);
       fprintf (fp, "th {\n");
       vftr_browse_make_html_indent (fp, 0, 2);
       fprintf (fp, "background-color: #dddddd;\n");
       vftr_browse_make_html_indent (fp, 0, 1);
       fprintf (fp, "}\n");
       fprintf (fp, "</style>\n");
       if (display_functions != NULL) {
          vftr_browse_print_navigation_bars (fp, display_functions, 0, n_display_functions, PROFILE);
       }
       
       return fp;
}

/**********************************************************************/

void vftr_browse_create_profile_header (FILE *fp) {
   fprintf (fp, "<h2>Runtime profile for rank %d</h2>\n", vftr_mpirank);
   fprintf (fp, "<table>\n");
   fprintf (fp, "<tr>\n");
   fprintf (fp, "<th>Calls</th>\n");
   fprintf (fp, "<th>Excl. time [s]</th>\n");
   fprintf (fp, "<th>Incl. time [s]</th>\n");
   fprintf (fp, "<th>%%abs</th>\n");
   fprintf (fp, "<th>%%cum</th>\n");
   fprintf (fp, "<th>Function</th>\n");
   fprintf (fp, "<th>Caller</th>\n");
   fprintf (fp, "<th>ID</th>\n");
   fprintf (fp, "</tr>\n");
}

/**********************************************************************/

void vftr_browse_table_cell_print (FILE *fp, column_t c, void *value_1, void *value_2) {
   switch (c.col_type) {
      case COL_INT:
         fprintf (fp, "<td> %*d </td>\n", c.n_chars, *(int*)value_1);
         break;
      case COL_DOUBLE:
         fprintf (fp, "<td> %*.*f </td>\n", c.n_chars, c.n_decimal_places, *(double*)value_1);
  	 break;
      case COL_CHAR_RIGHT:
         fprintf (fp, "<td> %*.s </td>\n", c.n_chars, (char*)value_1);
	 break;
   }
}

char *vftr_browse_table_cell_format_with_link (char *base_dir, char *func_name, char *base_format) {
   char format[256];
   snprintf (format, strlen(base_dir) + strlen(func_name) + strlen(base_format) + 27, "<td><a href=\"%s/%s\">%s</a></td>\n",
             base_dir, func_name, base_format);  
   return strdup (format);
}

void vftr_browse_print_table_line (FILE *fp, int stack_id,
				   long long application_runtime_usec, double sampling_overhead_time,
				   int n_calls, long long t_excl_usec, long long t_incl_usec, long long t_sum_usec, double t_overhead,
				   char *func_name, char *caller_name, column_t *prof_columns) {
	   int i_column = 0;
	   
	   fprintf (fp, "<tr>\n");

           double t_excl_sec = t_excl_usec * 1e-6;
           double t_incl_sec = t_incl_usec * 1e-6;
	   vftr_browse_table_cell_print (fp, prof_columns[i_column++], &n_calls, NULL);
    	   double value = n_calls > 0 ? t_excl_sec : 0;
           vftr_browse_table_cell_print (fp, prof_columns[i_column++], &value, NULL);
	   value = n_calls > 0 ? t_incl_sec : 0;
           vftr_browse_table_cell_print (fp, prof_columns[i_column++], &value, NULL);
           double t_part = (double)t_excl_usec / (double)application_runtime_usec * 100.0;
           vftr_browse_table_cell_print (fp, prof_columns[i_column++], &t_part, NULL);
           double t_cum = (double)t_sum_usec / (double)application_runtime_usec * 100.0;
           vftr_browse_table_cell_print (fp, prof_columns[i_column++], &t_cum, NULL);

	   if (vftr_environment.show_overhead->value) {
	      vftr_browse_table_cell_print (fp, prof_columns[i_column++], &t_overhead, NULL);
	      value = t_overhead / sampling_overhead_time * 100.0;
	      vftr_browse_table_cell_print (fp, prof_columns[i_column++], &value, NULL);
	      value = t_excl_sec > 0 ? t_overhead / t_excl_sec : 0.0;
	      vftr_browse_table_cell_print (fp, prof_columns[i_column++], &value, NULL);
	   }

	   if (vftr_events_enabled) {
  	      for (int i = 0; i < vftr_scenario_expr_n_formulas; i++) {
		 vftr_browse_table_cell_print (fp, prof_columns[i_column++], &vftr_scenario_expr_formulas[i].value, NULL);
	      }
	   }
	   
	   if (vftr_is_collective_mpi_function (func_name)) {
	      int n = strlen(func_name) + vftr_count_digits_int (vftr_mpirank) + 7;
	      char target_fun[n];
	      snprintf (target_fun, n, "%s_%d.html", func_name, vftr_mpirank);
    	      fprintf (fp, vftr_browse_table_cell_format_with_link (func_name, target_fun, " %s "), func_name);
	   } else {
	      vftr_browse_table_cell_print (fp, prof_columns[i_column++], func_name, NULL);
           }

	   vftr_browse_table_cell_print (fp, prof_columns[i_column++], caller_name, NULL);
	   vftr_browse_table_cell_print (fp, prof_columns[i_column++], &stack_id, NULL);

	   fprintf (fp, "</tr>\n");
}

/**********************************************************************/

void vftr_browse_finalize_table (FILE *fp) {
   fprintf (fp, "</table>\n");
   fclose (fp);
}

/**********************************************************************/

int vftr_browse_test_1 (FILE *fp_in, FILE *fp_out) {
	unsigned long long addrs[6];	
	function_t *func1 = vftr_new_function (NULL, "init", NULL, 0, false);
	function_t *func2 = vftr_new_function ((void*)addrs, "MAIN__", func1, 0, false);
	function_t *func3 = vftr_new_function ((void*)(addrs + 1), "A", func1, 0, false);
	function_t *func4 = vftr_new_function ((void*)(addrs + 2), "B", func1, 0, false);
	function_t *func5 = vftr_new_function ((void*)(addrs + 3), "C", func3, 0, false);
	function_t *func6 = vftr_new_function ((void*)(addrs + 4), "C", func4, 0, false);
	vftr_normalize_stacks();
	for (int i = 0; i < vftr_stackscount; i++) {
		vftr_func_table[i]->prof_current.calls = i + 1;
	}

	int *stack_indices, *func_indices;
	int n_indices;
	vftr_find_function_in_table ("C", &func_indices, &n_indices, false);	
	vftr_find_function_in_stack ("C", &stack_indices, &n_indices, false);
 	stack_leaf_t *stack_tree = NULL;
	for (int i = 0;  i < n_indices; i++) {
		//int n_functions_in_stack = vftr_stack_length (stack_indices[i]);
		// Why is n_functions_in_stack 2 instead of 3?
		int n_functions_in_stack = 3;
		int *stack_ids = (int*)malloc (n_functions_in_stack * sizeof(int));	
		int stack_id = stack_indices[i];
		int function_id = func_indices[i];
		for (int j = 0; j < n_functions_in_stack; j++) {
			stack_ids[j] = stack_id;
			stack_id = vftr_gStackinfo[stack_id].ret;
			printf ("%d ", stack_id);
		}
		printf ("\n");
		vftr_fill_into_stack_tree (&stack_tree, n_functions_in_stack, stack_ids, function_id);
		free (stack_ids);
	}
	long long dummy_l;
	double dummy_d;
	int dummy_i;
 	vftr_scan_stacktree (stack_tree->origin, 2, NULL, &dummy_d, &dummy_i, &dummy_d, &dummy_i, &dummy_i);
        display_function_t *display_functions[1];
	display_functions[0] = (display_function_t*)malloc (sizeof(display_function_t));
        display_functions[0]->func_name = "C";
        vftr_browse_print_stacktree_page (fp_out, false, display_functions, 0, 1, stack_tree->origin, NULL, 0.0, 1000, 1);
	free (stack_tree);
	return 0;
}

/**********************************************************************/
