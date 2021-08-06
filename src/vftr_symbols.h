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

#ifndef SYMBOLS_H
#define SYMBOLS_H

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <stdbool.h>
#include <elf.h>

extern FILE *vftr_log;

#define STRLEN 1000
#define LINESIZE 500

typedef struct {
    void *addr;
    char *name; /* Not de-mangled function name */
    int index; /* Section index */
} symtab_t;

typedef struct PathList {
    off_t            base;
    off_t          offset;
    char            *path;
    struct PathList *next;
} pathList_t;

extern int vftr_nsymbols;
extern symtab_t **vftr_symtab;

int vftr_cmpsym(const void *a, const void *b);

void vftr_print_symbol_table (FILE *f, bool include_addr);

void vftr_get_library_symtab (char *target, FILE *fp, off_t base, int pass);

typedef struct library_list {
    off_t base;
    off_t offset;
    char *path;
    struct library_list *next;
} library_list_t;

char *vftr_strip_module_name (char *base_name);

FILE *vftr_get_fmap ();

// Returns 1 if the creation of the symbol table fails.
// This can happen when symbol table offsets are found, but
// not supported by the VMAP_OFFSET option.
int vftr_create_symbol_table (int rank);

symtab_t **vftr_find_nearest(symtab_t **table, void *addr, int count);

char *vftr_find_symbol (void *addr);

char *vftr_demangle_cpp (char *m_name);

#endif
