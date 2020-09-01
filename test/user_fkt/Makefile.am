AM_CFLAGS = -O3 -finstrument-functions 
AM_CFLAGS += -I$(top_srcdir)/src/

AM_LDFLAGS = -L$(top_builddir)/src/.libs -lvftrace -lm

if MPI_PROFILING
  AM_CFLAGS += -D_MPI
endif
if USES_NEC_MPI
  AM_LDFLAGS += -mpiprof
endif


################
# System Tests #
################
check_PROGRAMS = cregions1 \
		 cregions2
		
cregions1_SOURCES = $(srcdir)/cregions1.c
cregions1_CFLAGS  = $(AM_CFLAGS)
cregions1_LDFLAGS = $(AM_LDFLAGS)

cregions2_SOURCES = $(srcdir)/cregions2.c
cregions2_CFLAGS  = $(AM_CFLAGS)
cregions2_LDFLAGS = $(AM_LDFLAGS)

dist_noinst_SCRIPTS = cregions1.sh \
		      cregions2.sh

TEST_EXTENSIONS = .sh
SH_LOG_COMPILER = $(SHELL)

TESTS = $(dist_noinst_SCRIPTS)
