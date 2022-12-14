#include <stdlib.h>
#include <stdio.h>

#include <string.h>
#include <elf.h>

#include "signal_handling.h"
#include "self_profile.h"

Elf64_Ehdr vftr_read_elf_header(FILE *fp) {
   SELF_PROFILE_START_FUNCTION;
   Elf64_Ehdr ElfHeader;
   size_t read_bytes = fread(&ElfHeader, 1, sizeof(Elf64_Ehdr), fp);
   if (read_bytes != sizeof(Elf64_Ehdr)) {
      perror("Reading Elf header");
      vftr_abort(0);
   }
   SELF_PROFILE_END_FUNCTION;
   return ElfHeader;
}

Elf64_Phdr *vftr_read_elf_program_header(FILE *fp, Elf64_Ehdr ElfHeader) {
   SELF_PROFILE_START_FUNCTION;
   Elf64_Phdr *ElfProgramHeader = NULL;
   int nProgramHeaderEntries = ElfHeader.e_phnum;
   size_t programHeaderSize = nProgramHeaderEntries*sizeof(Elf64_Phdr);
   ElfProgramHeader = (Elf64_Phdr*) malloc(programHeaderSize);
   fseek(fp, ElfHeader.e_phoff, SEEK_SET);
   size_t read_bytes = fread(ElfProgramHeader, 1,
                             programHeaderSize, fp);
   if (read_bytes != programHeaderSize) {
      perror("Reading Elf program headers");
      vftr_abort(0);
   }
   SELF_PROFILE_END_FUNCTION;
   return ElfProgramHeader;
}

Elf64_Shdr *vftr_read_elf_section_header(FILE *fp, Elf64_Ehdr ElfHeader) {
   SELF_PROFILE_START_FUNCTION;
   Elf64_Shdr *ElfSectionHeader = NULL;
   int nSectionHeaderEntries = ElfHeader.e_shnum;
   size_t sectionHeaderSize = nSectionHeaderEntries*sizeof(Elf64_Shdr);
   ElfSectionHeader = (Elf64_Shdr*) malloc(sectionHeaderSize);
   fseek(fp, ElfHeader.e_shoff, SEEK_SET);
   size_t read_bytes = fread(ElfSectionHeader, 1,
                             sectionHeaderSize, fp);
   if (read_bytes != sectionHeaderSize) {
      perror("Reading Elf section headers");
      vftr_abort(0);
   }
   SELF_PROFILE_END_FUNCTION;
   return ElfSectionHeader;
}

char *vftr_read_elf_header_string_table(FILE *fp, Elf64_Ehdr ElfHeader,
                                        Elf64_Shdr *ElfSectionHeader) {
   SELF_PROFILE_START_FUNCTION;
   size_t stringTableSize = ElfSectionHeader[ElfHeader.e_shstrndx].sh_size;
   if (stringTableSize == 0) {return NULL;}
   char *headerStringTable = (char*) malloc(stringTableSize);
   fseek(fp, ElfSectionHeader[ElfHeader.e_shstrndx].sh_offset, SEEK_SET);
   if (fread(headerStringTable, 1, stringTableSize, fp) != stringTableSize) {
      perror("Reading Elf header string table");
      vftr_abort(0);
   }
   SELF_PROFILE_END_FUNCTION;
   return headerStringTable;
}

int vftr_get_elf_string_table_index(char *headerStringTable, Elf64_Ehdr ElfHeader,
                                    Elf64_Shdr *ElfSectionHeader) {
   SELF_PROFILE_START_FUNCTION;
   int strtabidx = -1;
   for (int i=0; i<ElfHeader.e_shnum; i++) {
      char *name = headerStringTable + ElfSectionHeader[i].sh_name;
      if (!strcmp(name,".strtab")) {
         strtabidx = i;
         break;
      }
   }
   SELF_PROFILE_END_FUNCTION;
   return strtabidx;
}

int vftr_get_elf_symbol_table_index(char *headerStringTable, Elf64_Ehdr ElfHeader,
                                    Elf64_Shdr *ElfSectionHeader) {
   SELF_PROFILE_START_FUNCTION;
   int symtabidx = -1;
   for (int i=0; i<ElfHeader.e_shnum; i++) {
      char *name = headerStringTable + ElfSectionHeader[i].sh_name;
      if (!strcmp(name,".symtab")) {
         symtabidx = i;
         break;
      }
   }
   SELF_PROFILE_END_FUNCTION;
   return symtabidx;
}

char *vftr_read_elf_symbol_string_table(FILE *fp, Elf64_Shdr *ElfSectionHeader,
                                        int strtabidx) {
   SELF_PROFILE_START_FUNCTION;
   size_t strtabsize = ElfSectionHeader[strtabidx].sh_size;
   char *symbolStringTable = (char*) malloc(strtabsize);
   fseek(fp, ElfSectionHeader[strtabidx].sh_offset, SEEK_SET);
   size_t read_bytes = fread(symbolStringTable, 1, strtabsize, fp);
   if (read_bytes != strtabsize) {
      perror("Reading Elf symbol string table");
      vftr_abort(0);
   }
   SELF_PROFILE_END_FUNCTION;
   return symbolStringTable;
}

Elf64_Sym *vftr_read_elf_symbol_table(FILE *fp, Elf64_Shdr *ElfSectionHeader,
                                      int symtabidx) {
   SELF_PROFILE_START_FUNCTION;
   size_t symtabsize = ElfSectionHeader[symtabidx].sh_size;
   Elf64_Sym *symbolTable = (Elf64_Sym *)malloc(symtabsize);
   fseek(fp, ElfSectionHeader[symtabidx].sh_offset, SEEK_SET);
   size_t read_bytes = fread(symbolTable, 1, symtabsize, fp);
   if (read_bytes != symtabsize) {
      perror("Reading Elf symbol table");
      vftr_abort(0);
   }
   SELF_PROFILE_END_FUNCTION;
   return symbolTable;
}

void vftr_elf_symbol_table_count(Elf64_Shdr *ElfSectionHeader, int symtabidx,
                                 Elf64_Sym *symbolTable, int *symbolCount,
                                 int *validsymbolCount) {
   SELF_PROFILE_START_FUNCTION;
   int nsymb = ElfSectionHeader[symtabidx].sh_size/sizeof(Elf64_Sym);
   int validsymb = 0;
   for (int isymb=0; isymb<nsymb; isymb++) {
      Elf64_Sym s = symbolTable[isymb];
      if (ELF64_ST_TYPE(s.st_info) == STT_FUNC && s.st_value) {
         validsymb++;
      }
   }
   *symbolCount = nsymb;
   *validsymbolCount = validsymb;
   SELF_PROFILE_END_FUNCTION;
}
