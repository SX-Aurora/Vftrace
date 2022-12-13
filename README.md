# Vftrace

## About

Vftrace (visual ftrace) is a performance profiling library with a focus on applications in high-performance computing (HPC).
It is compatible with C, C++, and Fortran.
Vftrace produces an overview of the function calls appearing during an application's runtime and registers performance metrics such as the call number and the time spent in the code parts.
Multiprocess tracing of MPI-programs as well as the measurement of communication between processes is supported up to the MPI-3.1 standard.
Calls to CUDA kernels or CUDA library functions are traced using the CUpti profiling interface.

## Usage

Vftrace requires that your application has instrumendet function calls.
These are enabled with a compiler flag, most commonly named `-finstrument-functions`, as supported by the GNU, Intel, Clang, and NEC compilers.
To get access to the MPI functionality some MPI-implementations need extra flags to activate the internal profiling layer (e.g. NEC MPI needs -mpiprof).
After compiling, you must link your application against `libvftrace`, either statically or dynamically.
The application can then be run in the usual way.
In the default setting, a text file is created containing a runtime profile of the application.

## Download
You can clone the current version of Vftrace from github. The third party tools are included in the git repository as submodules, for your convinience.

```bash
git clone --recursive https://github.com/SX-Aurora/Vftrace.git
```
If you already cloned the repository without the --recursive flag you can get the submodules with
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
  - Clang
  - NEC

## Basic Principle

Vftrace uses the Cygnus function hooks:

```C
void __cyg_profile_func_enter (void *function_addr, void *caller_addr);
void __cyg_profile_func_exit  (void *function_addr, void *caller_addr);
```

These functions are used to intercept the program every time a function is called or returned from.
They need to be enabled by the compiler using the `-finstrument-functions` option. 
The arguments of these functions are the addresses of the symbols of the file mapped into virtual address space (`/proc/<pid>/maps`).
At initialization, Vftrace reads the ELF file of the executable, as well as its dependencies, to assign names to these symbols.
Vftrace dynamically grows an internal representation of the visited functions, their call history, and performance values.
Therefore, functions and their performance can be destinguished based on the callstack that called them.
This gives a detailed insight in where a program spends its time, thus enabling efficient optimization on the programmers side.

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
The Vftrace MPI wrappers record which ranks are communicating (who sends, who receives), the message size, message type, and communication time.
Non blocking communication is sampled by registering the non blocking call's request and checking for completion from time to time in the background.
Persistent requests are registered and handled like normal non blocking calls upon calling `MPI_Start` or `MPI_Startall`.
For collective communications inter- and intra-communicators are distinguished and handled to reflect their unique communication patterns.
Special buffers like `MPI_IN_PLACE` are taken care of and handled accordingly.

## Cupti (CUDA and other GPU programming models)

Vftrace allows the automatic profiling of CUDA calls using NVIDIA's CUPTI interface (https://docs.nvidia.com/cuda/cupti/index.html). This interface contains
callback functions for every function in the CUDA runtime. Vftrace registers these callback functions. The functions evoking these callbacks are
usually not instrumented and therefore separated from the default function hooks. Nevertheless, they are integrated into the Vftrace function stack tree and
appear in the profiles like all other functions.

A dedicated CUDA overview table displays the time spent in a CUDA function as well as the amount of data transferred between host and GPU (for cudaMemcpy etc.).
Currently, the CUPTI interface is only supported for one MPI process.

To enable CUDA profiling, configure with "--with-cupti=[cupti-dir]", where the argument is the path to the CUPTI installation, most commonly located at the same
location as the CUDA installation.

## User functions

Vftrace defines a few functions to enable programmers to modify their traced code in order to access internal Vftrace information during runtime.

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

### Note about Intel MPI
At the moment, Intel MPI does not support all PMPI symbols in the standard MPI library of oneAPI (as of version 2021.7.0). 
For reference, see https://community.intel.com/t5/Intel-oneAPI-HPC-Toolkit/More-PMPI-symbols-not-found/m-p/1348631#M9066.

The problem can be circumvented by including and linking the corresponding `mpi_f08` files. We have decided against this
in favor of waiting for Intel to solve this issue. If you want to use Intel MPI to profile a code which does not contain
MPI calls from Fortran, you can use `--disable-fortran` to prevent Vftrace from building the Fortran wrappers.

## Graphical User Interface

The graphical visualization tool for Vtrace profiles, `vfdviewer` is located at https://github.com/SpinTensor/vfdviewer.
It is written in C with the GTK+3 GUI.

## Authors

Vftrace was originally conceived by Jan Boerhout.
The main authors are:
  - Felix Uhl (felix.uhl@ruhr-uni-bochum.de)
  - Christian Weiss (christian.weiss@emea.nec.com)

## Third Party Tools

Vftrace uses the following open-source third party tools:

  - Adapted Jenkins (https://en.wikipedia.org/wiki/Jenkins_hash_function) and
    Murmur3 (https://en.wikipedia.org/wiki/MurmurHash) hash functions originally published under the creative common license (https://creativecommons.org/licenses/by-sa/3.0/).
    The hashes are used to identify individual stacks among different MPI-ranks.
  - cJSON: json parser from Dave Gamble (https://github.com/DaveGamble/cJSON).

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

Vftrace 2.0 is designed with support for OpenMP profiling in mind, relying on the callback functionality of OpenMP-5.x.
However, to our knowledge, there is currently no OpenMP implementation in which this callback system is functional. When these issues are resolved, the OpenMP support in Vftrace can easily be enabled.
