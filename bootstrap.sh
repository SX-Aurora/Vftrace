#!/usr/bin/bash

### This script deals preprocesses the Vftrace source directory for proper configuration.

### Download external dependencies (cjson, tinyexpr).
git submodule update --init

### Create configure
autoreconf -i

### libtool parses the output of "$CC -v" to determine which libraries need to be included in linker commands.
### It is known that e.g. some MPI wrappers output linker commands like "-l m -l gfortran", which is a correct linker line,
### but the libtool script needs to remove these white spaces. Otherwise, the linker lines with contain empty or mismatched "-l" statements,
### and linking will fail. Vftrace generates its own libtool script. Newer versions of autotools are supposed to have this issue fixed,
### but it has been observed that this is not always the case. The command below repairs and extends the checks that
### search for "-llib" patterns.
sed -i \
    -e 's/test x-L = "\$p"/test "\$p" = "-L"/' \
    -e 's/test x-R = "\$p"/test "\$p" = "-R"/' \
    -e 's/test "\$p" = "-R"; then/test "\$p" = "-R" ||\n          test "\$p" = "-l"; then/' \
    configure
