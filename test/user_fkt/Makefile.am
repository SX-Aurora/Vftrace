AM_CFLAGS = -O3 -finstrument-functions 
AM_CFLAGS += -I$(top_srcdir)/src/
AM_FCFLAGS = -O3 -finstrument-functions 
AM_FCFLAGS += -I$(top_builddir)/src/

AM_LDFLAGS = -L$(top_builddir)/src/.libs -lvftrace -lm

if MPI_PROFILING
  AM_CFLAGS += -D_MPI
  AM_FCFLAGS += -D_MPI
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

check_PROGRAMS += fregions1

fregions1_SOURCES = $(srcdir)/fregions1.F90
fregions1_CFLAGS  = $(AM_CFLAGS)
fregions1_LDFLAGS = $(AM_LDFLAGS)

dist_noinst_SCRIPTS += fregions1.sh

TEST_EXTENSIONS = .sh
SH_LOG_COMPILER = $(SHELL)

TESTS = $(dist_noinst_SCRIPTS)
