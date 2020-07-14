#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <elf.h>
#include <string.h>
#include "read_elf.h"

int read_maps () {
	char maps[80], line[LINESIZE];
	FILE *fmap;
	char *base;
	pid_t pid = getpid ();

	sprintf (maps, "/proc/%d/maps", pid);
	fmap = fopen(maps, "r");	
	fgets (line, LINESIZE, fmap);
	if (strstr (line, "check_offset") == NULL) {
		printf ("ERROR: program name not first in maps\n");
		return -1;
	}	
	base = strtok (line, "-");
	//printf ("base: %s\n", base);
	base_addr = strtoul (base, NULL, 16);
	//while (fgets (line, LINESIZE, fmap)) {
	//	printf ("%s\n", line);
	//}
	fclose (fmap);
	return 0;
}


int read_elf (char *this_fn) {	
	FILE *felf;

	Elf64_Ehdr ehdr;
	int nehdr = sizeof (Elf64_Ehdr);
	int i, n;
	int symstrIndex = -1;
	int symtabIndex = -1;
	int symbolCount;


	felf = fopen ("check_offset", "r");
	if (fread (&ehdr, 1, nehdr, felf) != (size_t)nehdr) {
		printf ("Coud not read ELF file!\n");
		return -1;
	}
	n = ehdr.e_shnum * sizeof (Elf64_Shdr);
	Elf64_Shdr *shdr = (Elf64_Shdr *) malloc(n);
	fseek (felf, (long)ehdr.e_shoff, SEEK_SET);
	if (fread(shdr, 1, n, felf) != (size_t)n) {
		printf ("Couldn't read section headers!\n");
		return -1;
	}

	
	int nst = shdr[ehdr.e_shstrndx].sh_size;
	char *headerStringTable = (char *) malloc(nst);
	memset (headerStringTable, 0, nst);
	fseek (felf, (long)shdr[ehdr.e_shstrndx].sh_offset, SEEK_SET);
	if (fread (headerStringTable, 1, nst, felf) != (size_t)nst) {
		printf ("Couldn't read string table!\n");
		return -1;
	}

	for (i = 0; i < ehdr.e_shnum; i++) {
		char *name = &headerStringTable[shdr[i].sh_name];
		if (!strcmp(name, ".strtab")) symstrIndex = i;
		else if (!strcmp(name, ".symtab")) symtabIndex = i;
	}

	int nsym;
	char *symbolStringTable;

        if (symstrIndex == -1) {
		printf ("No symbol string table!\n");
		return -1;
	} else {
		nsym = shdr[symstrIndex].sh_size;	
		symbolStringTable = (char *) malloc (nsym);
		memset (symbolStringTable, 0, nsym);
		fseek (felf, (long) shdr[symstrIndex].sh_offset, SEEK_SET);
		if (fread (symbolStringTable, 1, nsym, felf) != (size_t)nsym) {
			printf ("Couldn't read string table!\n");
			return -1;
		}
	}

	Elf64_Sym *symbolTable;

	if (symtabIndex == -1) {
		printf ("No symbol table!\n");
		return -1;
	} else {
		nsym = shdr[symtabIndex].sh_size;
		symbolTable = (Elf64_Sym*) malloc (nsym);
		memset (symbolTable, 0, nsym);
		symbolCount = nsym / sizeof (Elf64_Sym);
		fseek (felf, (long)shdr[symtabIndex].sh_offset, SEEK_SET);
		if (fread (symbolTable, 1, nsym, felf) != (size_t)nsym) {
			printf ("Couldn't read symbol table!\n");
			return -1;
		}
	}

	for (i = 0; i < symbolCount; i++) {
		Elf64_Sym s = symbolTable[i];
		if (ELF64_ST_TYPE (s.st_info) == STT_FUNC && s.st_value) {
			if (!strcmp (strdup(&symbolStringTable[s.st_name]), "test_1")) {
				symbol_addr = (unsigned long)s.st_value;
				enter_addr = (unsigned long)this_fn;	
			}	
		}
	}
       
  	fclose (felf);
}
