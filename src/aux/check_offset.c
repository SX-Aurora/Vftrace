#include <stdio.h>
#include <stdlib.h>
#include "read_elf.h"

void test_1 () {
}

int main () {
	test_1 ();
	if (symbol_addr == enter_addr) {
		printf ("OFFSET: NO\n");
		return 0;
	} else if (symbol_addr == enter_addr - base_addr) {
		printf ("OFFSET: YES\n");
		return 1;
	} else {
		printf ("ERROR: \n");
		printf ("base address: 0x%lx\n", base_addr);
		printf ("symbol address: 0x%lx\n", symbol_addr);
		printf ("enter address: 0x%lx\n", enter_addr);
		printf ("enter address - offset: 0x%lx\n", enter_addr - base_addr);
		return 0;
	}
}
