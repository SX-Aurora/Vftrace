if ENABLE_MPI
if USES_OPEN_MPI
   AM_TESTS_ENVIRONMENT=. $(top_srcdir)/test/environment/set_openmpi.sh;
else
if USES_NEC_MPI
if ON_VECTOR_ENGINE
   AM_TESTS_ENVIRONMENT=. $(top_srcdir)/test/environment/set_necmpi.sh;
else
   AM_TESTS_ENVIRONMENT=. $(top_srcdir)/test/environment/set_necvhmpi.sh;
endif
else
if USES_INTEL_MPI
   AM_TESTS_ENVIRONMENT=. $(top_srcdir)/test/environment/set_intelmpi.sh;
else
   AM_TESTS_ENVIRONMENT=. $(top_srcdir)/test/environment/set_genericmpi.sh;
endif
endif
endif
else
   AM_TESTS_ENVIRONMENT=. $(top_srcdir)/test/environment/set_nompi.sh;
endif

AM_TESTS_ENVIRONMENT+=. $(top_srcdir)/test/environment/unset_vftr_env.sh;

AM_CFLAGS = -O3 $(FINSTRUMENT_FLAG) 
AM_CFLAGS += -I$(top_srcdir)/src/
AM_FCFLAGS = -O3 $(FINSTRUMENT_FLAG) 
AM_FCFLAGS += -I$(top_builddir)/src/

AM_LDFLAGS = -L$(top_builddir)/src/.libs -lvftrace -lm

if ENABLE_MPI
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
if ENABLE_FORTRAN
if HAS_F_FINSTR
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
endif
endif

#############
# get stack #
#############
if ENABLE_FORTRAN
if HAS_F_FINSTR
check_PROGRAMS += fget_stack

fget_stack_SOURCES = $(srcdir)/fget_stack.F90
fget_stack_FCFLAGS = $(AM_FCFLAGS)
fget_stack_LDFLAGS = $(AM_LDFLAGS)

dist_noinst_SCRIPTS += fget_stack.sh
endif
endif

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

if ENABLE_FORTRAN
if HAS_F_FINSTR
check_PROGRAMS += fpause_resume

fpause_resume_SOURCES = $(srcdir)/fpause_resume.F90
fpause_resume_FCFLAGS = $(AM_FCFLAGS)
fpause_resume_LDFLAGS = $(AM_LDFLAGS)

dist_noinst_SCRIPTS += fpause_resume.sh
endif
endif

TEST_EXTENSIONS = .sh
SH_LOG_COMPILER = $(SHELL)

TESTS = $(dist_noinst_SCRIPTS)

CLEANFILES = vftrace_self_profile-* \
	     *.json
