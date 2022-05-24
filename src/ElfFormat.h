#ifndef ELF_FORMAT_H
#define ELF_FORMAT_H

#include <stdio.h>

#include <elf.h>

Elf64_Ehdr vftr_read_elf_header(FILE *fp);

Elf64_Phdr *vftr_read_elf_program_header(FILE *fp, Elf64_Ehdr ElfHeader);

Elf64_Shdr *vftr_read_elf_section_header(FILE *fp, Elf64_Ehdr ElfHeader);

char *vftr_read_elf_header_string_table(FILE *fp, Elf64_Ehdr ElfHeader,
                                        Elf64_Shdr *ElfSectionHeader);

int vftr_get_elf_string_table_index(char *headerStringTable, Elf64_Ehdr ElfHeader,
                                    Elf64_Shdr *ElfSectionHeader);

int vftr_get_elf_symbol_table_index(char *headerStringTable, Elf64_Ehdr ElfHeader,
                                    Elf64_Shdr *ElfSectionHeader);

char *vftr_read_elf_symbol_string_table(FILE *fp, Elf64_Shdr *ElfSectionHeader,
                                        int strtabidx);

Elf64_Sym *vftr_read_elf_symbol_table(FILE *fp, Elf64_Shdr *ElfSectionHeader,
                                      int symtabidx);

void vftr_elf_symbol_table_count(Elf64_Shdr *ElfSectionHeader, int symtabidx,
                                 Elf64_Sym *symbolTable, int *symbolCount,
                                 int *validsymbolCount);

#endif
