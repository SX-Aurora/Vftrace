#!/usr/bin/bash

git submodule update --init
autoreconf -i
sed -i \
    -e 's/test x-L = "\$p"/test "\$p" = "-L"/' \
    -e 's/test x-R = "\$p"/test "\$p" = "-R"/' \
    -e 's/test "\$p" = "-R"; then/test "\$p" = "-R" ||\n          test "\$p" = "-l"; then/' \
    configure
