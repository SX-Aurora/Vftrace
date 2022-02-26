#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#include <string.h>
#include <elf.h>

#include "address_type.h"
#include "symbols.h"
#include "ElfFormat.h"
#include "sorting.h"

library_t vftr_parse_maps_line(char *line) {
   library_t library = {
      .base = 0,
      .offset = 0,
      .path = NULL
   };
   // a line has the following format:
   // <baseaddr>-<topaddr> <permissions> <offset> <device> <inode> <libpath>
   // split the string by tokens
   char *base = strtok(line,"-");
   // skip over topaddr
   strtok(NULL," ");
   char *permissions = strtok(NULL," ");
   // only continue parsing if the library is marked executable
   if (permissions[2] != 'x') {return library;}
   char *offset = strtok(NULL, " ");
#ifndef __VMAP_OFFSET
   if (strncmp(offset, "00000000", 8)) {return library;}
#endif
   // skip over devide
   strtok(NULL," ");
   // skip over inode
   strtok(NULL," ");
   char *path = strtok(NULL," \n");
   // only continue if a valid path is found
   if (path == NULL || path[0] == '[') {return library;}
   // Filter out devices and system libraries
   // They are not instrumented and can be discarded
   // test for common path names
   if (!strncmp(path, "/dev/", 5)) {return library;}
   if (!strncmp(path, "/usr/lib", 8)) {return library;}
   if (!strncmp(path, "/usr/lib64", 10)) {return library;}
   if (!strncmp(path, "/lib", 4)) {return library;}
#ifdef __ve__
   if (!strncmp(path, "/opt/nec/ve/veos/lib", 20)) {return library;}
#endif
   sscanf(base, "%ld", &(library.base));
   library.path = strdup(path);
#ifdef __VMAP_OFFSET
   library.offset = strtoul(offset, NULL, 16);
#else
   library.offset = 0L;
#endif
   return library;
}

void vftr_print_library_list(FILE *fp, librarylist_t librarylist) {
   fprintf(fp, "Found libraries:\n");
   for (int ilib=0; ilib<librarylist.nlibraries; ilib++) {
      fprintf(fp, "%s (base=%lu, offset=%lu)\n",
              librarylist.libraries[ilib].path,
              librarylist.libraries[ilib].base,
              librarylist.libraries[ilib].offset);
   }
   fprintf(fp, "\n");
}

// read the different library paths from the applications map
librarylist_t vftr_read_library_maps() {
   char *fmappath = "/proc/self/maps";
   FILE *fmap = fopen(fmappath, "r");
   if (fmap == NULL) {
      perror (fmappath);
      abort();
   }

   librarylist_t librarylist = {
      .nlibraries = 0,
      .maxlibraries = 0,
      .libraries = NULL
   };

   char *lineptr = NULL;
   size_t buffsize = 0;
   ssize_t readbytes = 0;
   // read all lines
   while ((readbytes = getline(&lineptr, &buffsize, fmap)) > 0) {
      library_t library = vftr_parse_maps_line(lineptr);
      if (library.path != NULL) {
         // append it to the librarylist
         int idx = librarylist.nlibraries;
         librarylist.nlibraries++;
         if (librarylist.nlibraries > librarylist.maxlibraries) {
            librarylist.maxlibraries = 1.4*(librarylist.maxlibraries+1);
            librarylist.libraries = (library_t*)
               realloc(librarylist.libraries, librarylist.maxlibraries*sizeof(library_t));
         }
         librarylist.libraries[idx] = library;
      }
   }
   free(lineptr);
   fclose(fmap);
#ifdef _DEBUG
   vftr_print_library_list(stderr, librarylist);
#endif
   return librarylist;
}

void vftr_free_librarylist(librarylist_t *librarylist_ptr) {
   librarylist_t librarylist = *librarylist_ptr;
   if (librarylist.nlibraries > 0) {
      for (int ilib=0; ilib<librarylist.nlibraries; ilib++) {
         free(librarylist.libraries[ilib].path);
         librarylist.libraries[ilib].path = NULL;
      }
      free(librarylist.libraries);
      librarylist.libraries = NULL;
   }
}

void vftr_print_symbol_table(FILE *fp, symboltable_t symboltable) {
   for (unsigned int isym=0; isym<symboltable.nsymbols; isym++) {
      fprintf(fp, "%d 0x%llx %s\n",
              symboltable.symbols[isym].index,
              (unsigned long long int)
              symboltable.symbols[isym].addr,
              symboltable.symbols[isym].name);
   }
   fprintf(fp, "\n");
}

symboltable_t vftr_read_symbols() {
   // get all library paths that belong to the program
   librarylist_t librarylist = vftr_read_library_maps();

   symboltable_t symboltable = {
      .nsymbols = 0,
      .symbols = NULL,
   };
   // loop over all libraries and get their symbol tables
   for (int ilib=0; ilib<librarylist.nlibraries; ilib++) {
      FILE *fp = fopen(librarylist.libraries[ilib].path, "r");
      if (fp == NULL) {
         perror(librarylist.libraries[ilib].path);
         abort();
      }
      Elf64_Ehdr ElfHeader = vftr_read_elf_header(fp);
      Elf64_Shdr *ElfSectionHeader = vftr_read_elf_section_header(fp, ElfHeader);
      char *header_strtab = vftr_read_elf_header_string_table(fp, ElfHeader,
                                                              ElfSectionHeader);
      if (header_strtab == NULL) {
         free(ElfSectionHeader);
         fclose(fp);
         continue;
      }
      int strtabidx = vftr_get_elf_string_table_index(header_strtab,
                                                      ElfHeader,
                                                      ElfSectionHeader);
      if (strtabidx == -1) {
#ifdef _DEBUG
         fprintf(stderr, "No symbol string table in %s\n", librarylist.libraries[ilib].path);
#endif
         free(ElfSectionHeader);
         free(header_strtab);
         fclose(fp);
         continue;
      }
      char *stringtab = vftr_read_elf_symbol_string_table(fp, ElfSectionHeader,
                                                          strtabidx);
      int symtabidx = vftr_get_elf_symbol_table_index(header_strtab,
                                                      ElfHeader,
                                                      ElfSectionHeader);
      if (symtabidx == -1) {
#ifdef _DEBUG
         fprintf(stderr, "No symbol table in %s\n", librarylist.libraries[ilib].path);
#endif
         free(ElfSectionHeader);
         free(header_strtab);
         free(stringtab);
         fclose(fp);
         continue;
      }

      Elf64_Sym *ElfSymbolTable = vftr_read_elf_symbol_table(fp, ElfSectionHeader,
                                                             symtabidx);



      int symbolCount = 0;
      int validSymbolCount = 0;
      vftr_elf_symbol_table_count(ElfSectionHeader, symtabidx, ElfSymbolTable,
                                  &symbolCount, &validSymbolCount);
      if (validSymbolCount > 0) {
         // append the symbols from this library to the symboltable
         int nsymold = symboltable.nsymbols;
         int nsymnew = nsymold + validSymbolCount;
         symboltable.symbols = (symbol_t*)
            realloc(symboltable.symbols, nsymnew*sizeof(symbol_t));
         int copiedsymbols = 0;
         for (int isymb=0; isymb<symbolCount; isymb++) {
            Elf64_Sym s = ElfSymbolTable[isymb];
            if (ELF64_ST_TYPE(s.st_info) == STT_FUNC && s.st_value) {
               int jsymb = nsymold + copiedsymbols;
               copiedsymbols++;
#if defined(__ve__)
               symboltable.symbols[jsymb].addr =
                  (uintptr_t) (s.st_value);
#else
               off_t base = librarylist.libraries[ilib].base;
               symboltable.symbols[jsymb].addr =
                  (uintptr_t) (base + s.st_value);
#endif
               // Copy symbol name
               symboltable.symbols[jsymb].name = strdup(stringtab+s.st_name);
               symboltable.symbols[jsymb].index = s.st_shndx;
            }
         }
         symboltable.nsymbols = nsymnew;
      }

      free(ElfSectionHeader);
      free(header_strtab);
      free(stringtab);
      free(ElfSymbolTable);
      fclose(fp);
   }

   // sort the symbol table for faster access lateron with a binary search
   vftr_radixsort_symboltable(symboltable.nsymbols,
                              symboltable.symbols);

   vftr_free_librarylist(&librarylist);
#ifdef _DEBUG
   fprintf(stderr, "Elf-Symboltable:\n");
   vftr_print_symbol_table(stderr, symboltable);
#endif
   return symboltable;
}

void vftr_symboltable_free(symboltable_t *symboltable_ptr) {
   symboltable_t symboltable = *symboltable_ptr;
   if (symboltable.nsymbols > 0) {
      for (unsigned int isym=0; isym<symboltable.nsymbols; isym++) {
         free(symboltable.symbols[isym].name);
         symboltable.symbols[isym].name = NULL;
      }
      free(symboltable.symbols);
      symboltable.nsymbols = 0;
   }
}
