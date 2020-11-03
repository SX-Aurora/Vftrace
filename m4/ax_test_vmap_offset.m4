# ===========================================================================
#      https://www.gnu.org/software/autoconf-archive/ax_prog_cc_mpi.html
# ===========================================================================
#
# SYNOPSIS
#
#   AX_CHECK_VMAP_OFFSET
#
# DESCRIPTION
#
#   This macro tries to check if ELF symbols and internal function pointers
#   are offset relative to each other
#

AC_DEFUN([AX_CHECK_VMAP_OFFSET], [
   AC_PREREQ(2.50)
   AC_LANG(C)
   AC_MSG_CHECKING([if ELF-symbols need vmap offset])
   AC_RUN_IFELSE(
      [AC_LANG_SOURCE([[
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <elf.h>
#include <sys/types.h>

int testfunction() {return 0;};

off_t read_base_addr () {
#define LINESIZE 256
   char *base;
   char maps[LINESIZE];
   snprintf (maps, LINESIZE, "/proc/%d/maps", getpid());
   FILE *fmap = fopen(maps, "r");
   char line[LINESIZE];
   fgets(line, LINESIZE, fmap);
   base = strtok(line, "-");
   off_t base_addr = strtoul(base, NULL, 16);
   fclose (fmap);
   return base_addr;
}

off_t read_symbol_addr(char *exe_name) {
   FILE *elffile = fopen(exe_name, "r");
   // read elf-header
   Elf64_Ehdr elf_header;
   fread(&elf_header, sizeof(Elf64_Ehdr), 1, elffile);

   // allocate memory for section headers
   Elf64_Shdr *sec_header = (Elf64_Shdr*) malloc(elf_header.e_shnum*sizeof(Elf64_Shdr));
   // jump to begin of section header table
   fseek(elffile, elf_header.e_shoff, SEEK_SET);
   // read all section headers
   fread(sec_header, sizeof(Elf64_Shdr), elf_header.e_shnum, elffile);

   // get section header string table
   char *sec_headerstr = (char*) malloc(sec_header[elf_header.e_shstrndx].sh_size);
   fseek(elffile, sec_header[elf_header.e_shstrndx].sh_offset, SEEK_SET);
   fread(sec_headerstr, 1, sec_header[elf_header.e_shstrndx].sh_size, elffile);

   // get information about symbol table
   int sec_hdr_symtabidx = -1;
   int sec_hdr_symstridx = -1;
   for (int i=0; i<elf_header.e_shnum; i++) {
      if (!strcmp(".symtab", sec_headerstr+sec_header[i].sh_name)) {
         sec_hdr_symtabidx = i;
      } else if (!strcmp(".strtab", sec_headerstr+sec_header[i].sh_name)) {
         sec_hdr_symstridx = i;
      }
   }
   free(sec_headerstr);
   sec_headerstr = NULL;

   int nsymbols = sec_header[sec_hdr_symtabidx].sh_size/sizeof(Elf64_Sym);
   // allocate memory for symbol table
   Elf64_Sym *symbol_table = (Elf64_Sym*) malloc(nsymbols*sizeof(Elf64_Sym));
   char *symbolstr_table = (char*) malloc(sec_header[sec_hdr_symstridx].sh_size);
   // jump to symbol string table
   fseek(elffile, sec_header[sec_hdr_symstridx].sh_offset, SEEK_SET);
   fread(symbolstr_table, 1, sec_header[sec_hdr_symstridx].sh_size, elffile);
   // jump to symbol table
   fseek(elffile, sec_header[sec_hdr_symtabidx].sh_offset, SEEK_SET);
   // read symbol table
   fread(symbol_table, sizeof(Elf64_Sym), nsymbols, elffile);

   off_t testfunction_addr = -1;
   for (int i=0; i<nsymbols; i++) {
      if (symbol_table[i].st_name > 0) {
         int idx = symbol_table[i].st_name;
         if (!strcmp("testfunction", symbolstr_table+idx)) {
            testfunction_addr = symbol_table[i].st_value;
            break;
         }
      }
   }

   free(symbol_table);
   symbol_table = NULL;
   free(symbolstr_table);
   symbolstr_table = NULL;
   free(sec_header);
   sec_header = NULL;
   fclose(elffile);

   return testfunction_addr;
}

int main (int argc, char *argv[]) {
   if (argc <= 0) {printf("ERROR: \n"); return -2;}
   off_t base_addr = read_base_addr();
   off_t symbol_addr = read_symbol_addr(argv[0]);
   int (*testfunction_addr)() = testfunction;
   off_t off_testfunction_addr = (off_t) testfunction_addr;
//   printf("function addr = %p\n", testfunction_addr);
//   printf("base addr     = %p\n", (void*) base_addr);
//   printf("relative addr = %p\n", (void*) testfunction_addr - base_addr);
//   printf("symbol_addr   = %p\n", (void*) symbol_addr);
   if (symbol_addr == off_testfunction_addr) {
//      printf("OFFSET: NO\n");
      return 0;
   } else if (symbol_addr == off_testfunction_addr - base_addr) {
//      printf("OFFSET: YES\n");
      return 1;
   } else {
      printf("ERROR: \n");
      return -1;
   }
}
         ]])],
      [use_vmap_offset=no],
      [use_vmap_offset=yes])
   AM_CONDITIONAL([VMAP_OFFSET], [test "$use_vmap_offset" = "yes"])
   AC_MSG_RESULT([$use_vmap_offset])
])
