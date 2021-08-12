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

#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <limits.h>
#include <string.h>
#include <ctype.h>

#include "vftr_setup.h"

void vftr_rank0_printf (const char *fmt, ...) {
   if (vftr_mpirank == 0) printf (fmt);
}

/**********************************************************************/

int **vftr_ld_lookup;

// Create and free the Levenshtein lookup table
void vftr_init_ld_lookup (int n1, int n2) {
  vftr_ld_lookup = (int **)malloc (n1 * sizeof(int*));
  for (int i = 0; i < n1; i++) {
    vftr_ld_lookup[i] = (int*)malloc (n2 * sizeof(int));
  }
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      vftr_ld_lookup[i][j] = -1;
    }
  }
}

void vftr_free_ld_lookup (int n1) {
  for (int i = 0; i < n1; i++) {
    free(vftr_ld_lookup[i]);
  }
  free(vftr_ld_lookup);
}

/**********************************************************************/

// Compute the Levenshtein distance. The description of the algorithm is common, see e.g. Wikipedia.
int vftr_ld_kernel (char *a, char *b, int len_a, int len_b) {
  if (len_a == 0) {
    return len_b;
  } else if (len_b == 0) {
    return len_a;
  } else if (a[0] == b[0]) {
    if (vftr_ld_lookup[len_a - 1][len_b - 1] < 0) vftr_ld_lookup[len_a - 1][len_b - 1] = vftr_ld_kernel (a + 1, b + 1, len_a - 1, len_b - 1);
    return vftr_ld_lookup[len_a - 1][len_b - 1]; 
  } else { 
    int min = INT_MAX;
    if (vftr_ld_lookup[len_a - 1][len_b] < 0) vftr_ld_lookup[len_a - 1][len_b] = vftr_ld_kernel (a + 1, b, len_a - 1, len_b);
    int lev_1 = vftr_ld_lookup[len_a - 1][len_b];
    if (vftr_ld_lookup[len_a][len_b - 1] < 0) vftr_ld_lookup[len_a][len_b - 1] = vftr_ld_kernel (a, b + 1, len_a, len_b - 1);
    int lev_2 = vftr_ld_lookup[len_a][len_b - 1];  
    if (vftr_ld_lookup[len_a - 1][len_b - 1] < 0) vftr_ld_lookup[len_a - 1][len_b - 1] = vftr_ld_kernel (a + 1, b + 1, len_a - 1, len_b - 1);
    int lev_3 = vftr_ld_lookup[len_a - 1][len_b - 1];
    min = lev_1 < lev_2 ? lev_1 : lev_2;
    min = min < lev_3 ? min : lev_3;
    return 1 + min;
  }     
}

/**********************************************************************/

int vftr_levenshtein_distance (char *a, char *b) {
   int len_1 = strlen(a);
   int len_2 = strlen(b);
   vftr_init_ld_lookup (len_1, len_2);
   int ld = vftr_ld_kernel (a, b, len_1, len_2);
   vftr_free_ld_lookup (len_1);
   return ld;
}

/**********************************************************************/

void vftr_has_control_character (char *s, int *pos, int *char_num) {
  char *p = s;
  *pos = -1;
  if (char_num != NULL) *char_num = -1;
  int count = 0; 
  while (*p != '\0') {
    if (iscntrl(*p) && *p != '\n') {
      *pos = count;
      if (char_num != NULL) *char_num = *p;
      break;
    }
    count++;
    p++;
  }
}

/**********************************************************************/

bool vftr_string_is_number (char *s_check) {
  char *s = s_check;
  bool is_number = true;
  while (*s != '\0') {
     is_number &= isdigit(*s);
     s++;
  }
}
/**********************************************************************/


