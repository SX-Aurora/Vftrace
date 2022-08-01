#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>

#include <string.h>
#include <elf.h>
#include <ctype.h>
#ifdef _LIBERTY
#include <demangle.h>
#endif

#include "realloc_consts.h"
#include "custom_types.h"
#include "symbols.h"
#include "ElfFormat.h"
#include "search.h"
#include "sorting.h"
#include "regular_expressions.h"
#include "misc_utils.h"

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
   // skip over device
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
   library.base = strtoul(base, NULL, 16);
   library.path = strdup(path);
   library.offset = strtoul(offset, NULL, 16);
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
            librarylist.maxlibraries = librarylist.maxlibraries*vftr_realloc_rate+vftr_realloc_add;
            librarylist.libraries = (library_t*)
               realloc(librarylist.libraries, librarylist.maxlibraries*sizeof(library_t));
         }
         librarylist.libraries[idx] = library;
      }
   }
   free(lineptr);
   fclose(fmap);
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
      fprintf(fp, "%d 0x%llx %s%s\n",
              symboltable.symbols[isym].index,
              (unsigned long long int)
              symboltable.symbols[isym].addr,
              symboltable.symbols[isym].name,
              symboltable.symbols[isym].precise ? "*" : "");
   }
   fprintf(fp, "\n");
}

// merge two previously sorted symbol tables into one
symboltable_t vftr_merge_symbol_tables(symboltable_t symtabA, symboltable_t symtabB) {
   symboltable_t symboltable;
   symboltable.nsymbols = symtabA.nsymbols + symtabB.nsymbols;
   symboltable.symbols = (symbol_t*) malloc(symboltable.nsymbols*sizeof(symbol_t));

   unsigned int idxA = 0;
   unsigned int idxB = 0;
   unsigned int idxS = 0;

   while (idxA < symtabA.nsymbols && idxB < symtabB.nsymbols) {
      if (symtabA.symbols[idxA].addr <= symtabB.symbols[idxB].addr) {
         symboltable.symbols[idxS] = symtabA.symbols[idxA];
         idxA++;
      } else {
         symboltable.symbols[idxS] = symtabB.symbols[idxB];
         idxB++;
      }
      idxS++;
   }

   while (idxA < symtabA.nsymbols) {
      symboltable.symbols[idxS] = symtabA.symbols[idxA];
      idxA++;
      idxS++;
   }

   while (idxB < symtabB.nsymbols) {
      symboltable.symbols[idxS] = symtabB.symbols[idxB];
      idxA++;
      idxS++;
   }

   return symboltable;
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
      Elf64_Phdr *ElfProgramHeader =
         vftr_read_elf_program_header(fp, ElfHeader);
      // Offset of the segment in the file image.
      Elf64_Off elf_offset = 0;
      // Virtual address of the segment in memory.
      Elf64_Addr elf_vaddr = 0;
      for (int iheader=0; iheader<ElfHeader.e_phnum; iheader++) {
         // We only care for the segment containing the program header itself
         if ((ElfProgramHeader+iheader)->p_type == 0x00000006) {
            elf_offset = (ElfProgramHeader+iheader)->p_offset;
            elf_vaddr = (ElfProgramHeader+iheader)->p_vaddr;
            break;
         }
      }
      Elf64_Shdr *ElfSectionHeader = vftr_read_elf_section_header(fp, ElfHeader);
      char *header_strtab = vftr_read_elf_header_string_table(fp, ElfHeader,
                                                              ElfSectionHeader);
      if (header_strtab == NULL) {
         free(ElfProgramHeader);
         free(ElfSectionHeader);
         fclose(fp);
         continue;
      }
      int strtabidx = vftr_get_elf_string_table_index(header_strtab,
                                                      ElfHeader,
                                                      ElfSectionHeader);
      if (strtabidx == -1) {
         free(ElfProgramHeader);
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
         free(ElfProgramHeader);
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
               // apply all offsets to the addresses
               // in order to match them with the called addresses
               off_t lbase = librarylist.libraries[ilib].base;
               off_t loffset = librarylist.libraries[ilib].offset;
               symboltable.symbols[jsymb].addr =
                  (uintptr_t) (s.st_value + lbase - loffset + elf_offset - elf_vaddr);
               // Copy symbol name
               symboltable.symbols[jsymb].name = strdup(stringtab+s.st_name);
               symboltable.symbols[jsymb].index = s.st_shndx;
               // remove trailing fortran underscore
               vftr_chop_trailing_char(symboltable.symbols[jsymb].name, '_');

               // set precise value to default: false
               symboltable.symbols[jsymb].precise = false;
            }
         }
         symboltable.nsymbols = nsymnew;
      }

      free(ElfProgramHeader);
      free(ElfSectionHeader);
      free(header_strtab);
      free(stringtab);
      free(ElfSymbolTable);
      fclose(fp);
   }

   // sort the symbol table for faster access lateron with a binary search
   vftr_sort_symboltable(symboltable.nsymbols,
                         symboltable.symbols);

   vftr_free_librarylist(&librarylist);
   return symboltable;
}

void vftr_symboltable_determine_preciseness(symboltable_t *symboltable_ptr,
                                            regex_t *preciseregex) {
#ifdef _MPI
   // regex to check for MPI functions
   // Case insensitivity is needed for Fortran
   regex_t *mpi_regex = vftr_compile_regexp("^(MPI|mpi)_[A-Za-z]*");
#endif
   // regex to check for the vftrace internal pause/resume functions
   regex_t *pause_regex = vftr_compile_regexp("^vftrace_(pause|resume)$");
   for (unsigned int isym=0; isym<symboltable_ptr->nsymbols; isym++) {
      char *name = symboltable_ptr->symbols[isym].name;
      bool precise;
      precise = vftr_pattern_match(preciseregex, name);
#ifdef _MPI
      precise = precise || vftr_pattern_match(mpi_regex, name);
#endif
      precise = precise || vftr_pattern_match(pause_regex, name);
      symboltable_ptr->symbols[isym].precise = precise;
   }

#ifdef _MPI
   regfree(mpi_regex);
   free(mpi_regex);
#endif
   regfree(pause_regex);
   free(pause_regex);
}

void vftr_symboltable_strip_fortran_module_name(symboltable_t *symboltable_ptr,
                                                bool strip_module_names) {
   if (strip_module_names) {
      int nmodule_delimiters = 3;
      char *module_delimiters[] = {
         "_MOD_", // gfortran
         "_MP_",  // nfort
         "_mp_"  // ifort
      };

      for (unsigned int isym=0; isym<symboltable_ptr->nsymbols; isym++) {
         char *name = symboltable_ptr->symbols[isym].name;
         for (int idelim=0; idelim<nmodule_delimiters; idelim++) {
            vftr_trim_left_with_delimiter(name, module_delimiters[idelim]);
         }
      }
   }
}

#ifdef _LIBERTY

void vftr_has_control_character (char *s, int *pos, int *char_num) {
   char *p = s;
   *pos = -1;
   if (char_num != NULL) {
      *char_num = -1;
   }
   int count = 0;
   while (*p != '\0') {
      if (iscntrl(*p) && *p != '\n') {
         *pos = count;
         if (char_num != NULL) {
            *char_num = *p;
         }
         break;
       }
      count++;
      p++;
   }
}

char *vftr_demangle_cxx(char *name) {
   char *demangled_name = cplus_demangle(name, 0);

   if (demangled_name == NULL) {
      // Not a C++ symbol
      return strdup(name);
   }
 
   // The output of cplus_demangle has been observed to sometimes include
   // non-ASCII control characters. These can lead to problems later in Vftrace.
   // Therefore, the demangled name is ignored.
   int has_control_char;
   vftr_has_control_character (demangled_name, &has_control_char, NULL);
   if (has_control_char >= 0) {
      free(demangled_name);
      return strdup(name); 
   }
   
   // demangled symbols contain <type> strings
   char *ptr = demangled_name;
   while (*ptr != '\0') {
      if (*ptr == '<') {
         *ptr = '\0';
      } else {
         ptr++;
      }
   }
   
   return demangled_name;
}

void vftr_symboltable_demangle_cxx_name(symboltable_t *symboltable_ptr,
                                        bool demangle_cxx) {
   if (demangle_cxx) {
      for (unsigned int isym=0; isym<symboltable_ptr->nsymbols; isym++) {
         char *name = symboltable_ptr->symbols[isym].name;
         char *demang_name = vftr_demangle_cxx(name);
         free(symboltable_ptr->symbols[isym].name);
         symboltable_ptr->symbols[isym].name = demang_name;
      }
   }
}
#endif

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

int vftr_get_symbID_from_address(symboltable_t symboltable,
                                 uintptr_t address) {
   return vftr_binary_search_symboltable(symboltable.nsymbols,
                                         symboltable.symbols,
                                         address);
}

char *vftr_get_name_from_address(symboltable_t symboltable,
                                 uintptr_t address) {
   int symbID = vftr_get_symbID_from_address(symboltable,
                                             address);
   if (symbID >= 0) {
      return symboltable.symbols[symbID].name;
   } else {
      return "(UnknownFunctionName)";
   }
}

char *vftr_get_name_from_symbID(symboltable_t symboltable,
                                int symbID) {
   if (symbID >= 0) {
      return symboltable.symbols[symbID].name;
   } else {
      return "(UnknownFunctionName)";
   }
}

bool vftr_get_preciseness_from_symbID(symboltable_t symboltable,
                                      int symbID) {
   return symboltable.symbols[symbID].precise;
}

bool vftr_get_preciseness_from_address(symboltable_t symboltable,
                                       uintptr_t address) {
   int symbID = vftr_get_symbID_from_address(symboltable,
                                             address);
   return vftr_get_preciseness_from_symbID(symboltable, symbID);
}
