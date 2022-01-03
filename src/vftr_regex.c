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
#include <regex.h>

#include "vftr_symbols.h"

/**********************************************************************/

regex_t *vftr_compile_regexp(char *pattern) {
    int err;
    regex_t *r;
    r = (regex_t*) malloc(sizeof(regex_t));
    if ((err = regcomp (r, pattern, REG_NOSUB|REG_EXTENDED))) {
        char msg[256];
        fprintf(stderr, "Vftrace: Invalid Regular Expression (%s): %s\n", pattern, msg);
    }
    return err ? NULL : r;
}

/**********************************************************************/

bool vftr_pattern_match(regex_t *r, char *s) {
    if (r != NULL) {
       regmatch_t regmatch[8];
       return !regexec (r, s, 0, regmatch, 0);
    } else {
       return false;
    }
}
