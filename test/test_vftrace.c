int this_fails () {
	return 1;
}

int this_passes () {
	return 0;
}

int main (int argc, char **argv) {
	if (!strcmp (argv[1], "this_fails")) {
		return this_fails();
	} else if (!strcmp (argv[1], "this_passes")) {
		return this_passes();
	}
}
