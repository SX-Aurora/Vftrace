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


###########
# regions #
###########
check_PROGRAMS = cregions1 \
		 cregions2 \
		 cregions3 \
		 cregions4
		
cregions1_SOURCES = $(srcdir)/cregions1.c
cregions1_CFLAGS  = $(AM_CFLAGS)
cregions1_LDFLAGS = $(AM_LDFLAGS)

cregions2_SOURCES = $(srcdir)/cregions2.c
cregions2_CFLAGS  = $(AM_CFLAGS)
cregions2_LDFLAGS = $(AM_LDFLAGS)

cregions3_SOURCES = $(srcdir)/cregions3.c
cregions3_CFLAGS  = $(AM_CFLAGS)
cregions3_LDFLAGS = $(AM_LDFLAGS)

cregions4_SOURCES = $(srcdir)/cregions4.c
cregions4_CFLAGS  = $(AM_CFLAGS)
cregions4_LDFLAGS = $(AM_LDFLAGS)

dist_noinst_SCRIPTS = cregions1.sh \
		      cregions2.sh \
		      cregions3.sh \
		      cregions4.sh

check_PROGRAMS += fregions1 \
		  fregions2 \
		  fregions3 \
		  fregions4

fregions1_SOURCES = $(srcdir)/fregions1.F90
fregions1_FCFLAGS = $(AM_FCFLAGS)
fregions1_LDFLAGS = $(AM_LDFLAGS)

fregions2_SOURCES = $(srcdir)/fregions2.F90
fregions2_FCFLAGS = $(AM_FCFLAGS)
fregions2_LDFLAGS = $(AM_LDFLAGS)

fregions3_SOURCES = $(srcdir)/fregions3.F90
fregions3_FCFLAGS = $(AM_FCFLAGS)
fregions3_LDFLAGS = $(AM_LDFLAGS)

fregions4_SOURCES = $(srcdir)/fregions4.F90
fregions4_FCFLAGS = $(AM_FCFLAGS)
fregions4_LDFLAGS = $(AM_LDFLAGS)

dist_noinst_SCRIPTS += fregions1.sh \
		       fregions2.sh \
		       fregions3.sh \
		       fregions4.sh

#############
# get stack #
#############
check_PROGRAMS += cget_stack

cget_stack_SOURCES = $(srcdir)/cget_stack.c
cget_stack_CFLAGS  = $(AM_CFLAGS)
cget_stack_LDFLAGS = $(AM_LDFLAGS)

dist_noinst_SCRIPTS += cget_stack.sh

################
# pause/resume #
################
check_PROGRAMS += cpause_resume

cpause_resume_SOURCES = $(srcdir)/cpause_resume.c
cpause_resume_CFLAGS  = $(AM_CFLAGS)
cpause_resume_LDFLAGS = $(AM_LDFLAGS)

dist_noinst_SCRIPTS += cpause_resume.sh

check_PROGRAMS += fpause_resume

fpause_resume_SOURCES = $(srcdir)/fpause_resume.F90
fpause_resume_FCFLAGS = $(AM_FCFLAGS)
fpause_resume_LDFLAGS = $(AM_LDFLAGS)

dist_noinst_SCRIPTS += fpause_resume.sh

TEST_EXTENSIONS = .sh
SH_LOG_COMPILER = $(SHELL)

TESTS = $(dist_noinst_SCRIPTS)
