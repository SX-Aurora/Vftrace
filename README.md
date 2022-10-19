# Vftrace

## About

Vftrace (visual ftrace) is a performance profiling library with a focus on applications in high-performance computing (HPC).
It is compatible with C, C++, and Fortran.
Multiporocess tracing of MPI-programs as well as the measurement of communication between processes is supported up to the MPI-3.1 standard.
Calls to CUDA are traced using the CUPTI library.
Vftrace produces an overview of the function calls appearing during an application's runtime and registers the call number and the time spent in the code parts.

## Usage

Vftrace requires that your application has instrumendet function calls.
These are enabled with a compiler flag, most commonly known as `-finstrument-functions`, as supported by the GNU, Intel, and NEC compilers.
To get access to the MPI functionality some MPI-implementations need extra flags to activate the internal profiling layer (e.g. NEC MPI needs -mpiprof).
After compiling, you must link your application against `libvftrace`, either statically or dynamically.
The application can then be run in the usual way.
In the default setting, a text file is created containing a run-time profile of the application.

## Download
You can clone the current version of the vftrace from github.
The third party tools are included in the git repository as submodules, for your convinience.
```bash
git clone --recursive https://github.com/SX-Aurora/Vftrace.git
```
If you already cloned the repository without the `--recursive` flag you can get the submodules with
```bash
git submodule update --init
```

## Prerequisites & Installation 

Vftrace is written in C.
For the Fortran interface, there is also some Fortran code.
Vftrace is built using the standard autotools toolchain.
For more information, consult the `INSTALL` file included in this repository.

We recommend to compile your application with the same compiler you used to compile Vftrace.
It has to support function instrumentation.
To our knowledge, this is given for the following list of compilers:
  - GNU
  - Intel
  - NEC

## Basic Principle

Vftrace uses the Cygnus function hooks:

```C
   void __cyg_profile_func_enter (void *function_addr, void *caller_addr);
   void __cyg_profile_func_exit  (void *function_addr, void *caller_addr);
```

These functions are used to intercept the program every time a function is called or returned from one.
They need to be enabled by the compiler using the `-finstrument-functions` option. 
The arguments of these functions are the addresses of the symbols of the file mapped into virtual address space (`/proc/<pid>/maps`).
At initialization, Vftrace reads the ELF file of the executable, as well as its dependencies, to assign names to these symbols.
Vftrace dynamically grows an internal representation of the visited functions, their call history, and performance values.
Therefore, functions and their performance can be destinguished based on the callstack that called them.
This gives a detailed insight in where a program spends its time, thus enableing efficient optimization on the programmers side.

## MPI Profiling

Vftrace has wrappers to MPI function calls.
These wrappers are instrumented and call a `vftr_MPI` version of the function.
The wrappers for C and Fortran both call the same `vftr_MPI` routine to reduce code duplication and enable easier maintenance.
```C
   int MPI_Send(const void *buf, int count, MPI_Datatype datatype,
                int dest, int tag, MPI_Comm comm) {
      return vftr_MPI_Send(buf, count, datatype, dest, tag, comm);
   }
```

In these `vftr_MPI`-functions the communication is executed by calling the corresponding `PMPI` function:
```C
   int vftr_MPI_Send(const void *buf, int count, MPI_Datatype datatype,
                     int dest, int tag, MPI_Comm comm) {
   
      // disable profiling based on the Pcontrol level
      if (vftrace_Pcontrol_level == 0) {
         return PMPI_Send(buf, count, datatype, dest, tag, comm);
      } else {
         long long tstart = vftr_get_runtime_nsec();
         int retVal = PMPI_Send(buf, count, datatype, dest, tag, comm);
         long long tend = vftr_get_runtime_nsec();
   
         vftr_store_sync_message_info(send, count, datatype, dest, tag, comm, tstart, tend);
   
         return retVal;
      }
   }
```
The `PMPI_` symbols do the same as their `MPI_` counterpart and are part of any standard complying MPI-implementation.
This way, the MPI functions as used by the application are instrumented.
The functionality inside the wrapper enables in-depth MPI sampling.
The vftrace MPI-wrapper record which ranks are communicating (who sends, who receives), the message size, message type, and communication time.
Non blocking communication is sampled by registering the non blocking call's request and checking for completion from time to time in the background.
Persistent requests are registered and handled like normal non blocking calls upon calling `MPI_Start` or `MPI_Startall`.
For collective communications inter- and intra-communicators are distinguished and handled to reflect their unique communication patterns.
Special buffers like `MPI_IN_PLACE` are taken care of and handled accordingly.

## Cupti (CUDA and other GPU programming models)

## User functions

Vftrace defines a few functions to enable programmers to modify their traced code in order to access vftrace internal information during runtime.

### vftrace_region_begin, vftrace_region_end
   If you encounter a time consuming routine in your profiled code, which does multiple things and you need to identify which portion of the routine is at fault, the vftrace regions are the tool to do this.
   They define  the  start and end of a region in the code, which should be monitored independently from from a function entry.
   The functions take as an  argument a  unique  string  identifier.
   The defined region appears in the logfile and vfd files under the this name.

   Example in C:
```C
void testfunction() {
   ...
   vftrace_region_begin("NameOfTheRegion");
   // code to be profiled independently
   ...
   // from the rest of the function
   vftrace_region_end("NameOfTheRegion");
   ...
}
```
   Example in Fortran:
```Fortran
      SUBROUTINE testroutine()
         ...
         CALL vftrace_region_begin("NameOfTheRegion")
         ! code to be profiled independently
         ...
         ! from the rest of the routine
         CALL vftrace_region_end("NameOfTheRegion")
         ...
      END SUBROUTINE
```
### vftrace_get_stack
   If you want to know which callpath your program took in order to arrive at a specific routine you may utilize `vftrace_get_stack`.
   The function returns a string containing the callstack to the current function. (In order to avoid memory leaks, this string needs to be freed afterwards.)

   Example in C:
```C
   printf("%s\n", vftrace_get_stack());
```
   Example in Fortran:
```Fortran
   write(*,*) vftrace_get_stack()
```

### vftrace_pause, vftrace_resume
   If your code contains portions that you do not whish to profile, maybe an initialization, and which would render your profile cluttered, you can use `vftrace_pause` and `vftrace_resume` to halt profiling for portions of your program execution.

   Example in C:
```C
   int main() {
      // This code is profiled
      ...
      vftrace_pause();
      // This code is not profiled
      ...
      vftrace_resume();
      // This code is profiled again
      ...
   }
```
   Example in Fortran:
```Fortran
   PROGRAM testprogram
      ! This code is profiled
      ...
      CALL vftrace_pause()
      ! This code is not profiled
      ...
      CALL vftrace_resume()
      ! This code is code profiled again
      ...
   END PROGRAM testprogram
```

## Graphical User Interface

The graphical visualization tool for Vtrace profiles, Vfview, is located at https://github.com/SX-Aurora/Vfview.

With the environment variable `VFTRACE_CREATE_HTML`, Vftrace produces a graphical visualization of profiling results as HTML output.
In the application directory, a directory called "browse" is created.
In there is an index.html, which can be opened with
a usual web browser.
It allows for navigation between different views, MPI ranks and MPI collective functions.You can either download the entire browse directory to your local machine, or access it remotely (given suitable network configurations).

## Authors

Vftrace was originally conceived by Jan Boerhout.
The main authors are:
  - Felix Uhl (felix.uhl@emea.nec.com)
  - Christian Weiss (christian.weiss@emea.nec.com)

## Third Party Tools

Vftrace uses the following open-source third party tools:

  - The json parser "jsmn" by Serge Zaitsev (https://github.com/zserge/jsmn).
  It is used to read in the hardware scenario files.
  - Lewis van Winkle's "tinyexpr" (https://github.com/codeplea/tinyexpr).
  It is used to parse the formula strings which define hardware observables.
  - Adapted Jenkins (https://en.wikipedia.org/wiki/Jenkins_hash_function) and
    Murmur3 (https://en.wikipedia.org/wiki/MurmurHash) hash functions originally published under the creative common license (https://creativecommons.org/licenses/by-sa/3.0/).
    The hashes are used to identify individual stacks among different MPI-ranks.

## Licensing

Vftrace is licensed under The GNU general public license (GPL), which means that you are free to copy and modify the source code under the condition that you keep the license.

## How to Contribute

You are free to clone or fork this repository as you like.
If you wish to make a contribution, please open up a pull request.
Consult the `CODEOWNERS` file for more information about contact persons for specific parts of the code.
If you find a bug in Vftrace or have an idea for an improvement, please submit an issue on github.

## FAQ

### Which languages are supported besides C and Fortran?

In principle, Vftrace can support every compiled language which can create a function hook like `__cyg_profile_func_enter`.
As of now, the only languages we know of that support this feature are
C, C++ and Fortran.
On top of that, the executable has to be in the ELF format and Vftrace must be able to parse the symbol names out of it.
This is still not optimal for C++, which is the reason why it is not yet officially supported.

### Does Vftrace support OpenMP?

Vftrace does not yet support OpenMP.
This is because it is still an open question how the dynamic creation of threads can be combined with the function-stack structure of Vfrace.
