#!/usr/bin/env sh

# Expect prefix and include dir
DESTDIR="${1}/${2}"

VFTRACE_MOD="$(find ${MESON_BUILD_ROOT} -name 'vftrace.mod')"
cp -a "${VFTRACE_MOD}" "${DESTDIR}"
