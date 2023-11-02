#include <Python.h>
#include "frameobject.h"
#include "structmember.h"

#include <stdio.h>

#include "thread_types.h"
#include "threadstack_types.h"
#include "stack_types.h"
#include "profiling_types.h"
#include "profiling.h"
#include "callprofiling.h"
#include "vftrace_state.h"
#include "threadstacks.h"
#include "vftr_initialize.h"
#include "timer.h"
#include "hashing.h"
#include "misc_utils.h"


static PyObject *init_vftrace (PyObject *self);

static PyMethodDef vftraceMethods[] = {
   {
      "startVftrace", (PyCFunction)init_vftrace, METH_NOARGS,
      "start Vftrace",
   },
   {NULL, NULL, 0, NULL}
};

static const struct {
   int event;
   const char *callback_method;
} callback_table[] = {
  //{PY_MONITORING_EVENT_PY_START, "_pystart_callback"},
  //{PY_MONITORING_EVENT_PY_RETURN, "_pyreturn_callback"},
  {0, NULL}
};

const char *repr (PyObject *o) {
   PyObject *r = PyObject_Repr(o);
   PyObject *s = PyUnicode_AsEncodedString(r, "utf-8", "~E~");
   char *out = PyBytes_AS_STRING(s);
   Py_XDECREF(r);
   Py_XDECREF(s);
   return strdup(out);
}

static int profiler_callback (PyObject *self, PyFrameObject *frame, int what, PyObject *args) {
   long long function_time_begin = vftr_get_runtime_nsec();
   char *func_name;
   uint64_t pseudo_addr;
   if (what ==  PyTrace_CALL || what == PyTrace_RETURN) {
      PyCodeObject *fn = (PyCFunctionObject*)frame->f_code;
      //func_name = repr(fn->co_name);
      //printf ("NULL? %d\n", func_name == NULL);
      //fflush(stdout);
      //printf ("strlen: %d\n", strlen(func_name));
      //fflush(stdout);
      //printf ("str: <<<");
      //for (int i = 0; i < strlen(func_name); i++) {
      //   printf ("%c", func_name[i]);
      //}
      //printf (">>>\n");
      //printf ("strfull: <<<%s>>>\n", func_name);
      //fflush(stdout);
      PyObject *r = PyObject_Repr(fn->co_name);
      PyObject *s = PyUnicode_AsEncodedString(r, "utf-8", "~E~");
      func_name = strdup(PyBytes_AS_STRING(s));
      Py_XDECREF(r);
      Py_XDECREF(s);
      r = PyObject_Repr(fn->co_filename);
      s = PyUnicode_AsEncodedString(r, "utf-8", "~E~");
      char *file_name = strdup(PyBytes_AS_STRING(s));
      Py_XDECREF(r);
      Py_XDECREF(s);
      //pseudo_addr = (uint64_t)strlen(func_name);
      //printf ("%s / %s\n", file_name, func_name);
      char *buf = (char*)malloc ((strlen(func_name) + strlen(file_name) + 1) * sizeof(char));
      strcpy (buf, func_name);
      strcat (buf, file_name);
      pseudo_addr = vftr_jenkins_murmur_64_hash (strlen(func_name), (uint8_t*)func_name);
      pseudo_addr += vftr_jenkins_murmur_64_hash (strlen(file_name), (uint8_t*)file_name);
      //pseudo_addr = vftr_jenkins_murmur_64_hash (strlen(buf), (uint8_t*)buf);
      //pseudo_addr += (uint64_t)frame->f_back;
      pseudo_addr += (uint64_t)frame->f_lineno;
      int lineno = fn->co_firstlineno;
      printf ("%s: %s @ %s %d 0x%lx -> 0x%lx\n", what == PyTrace_CALL ? "Entry" : "Exit", func_name, file_name, lineno, (uint64_t)frame->f_back, pseudo_addr);
      free (buf);
   } else if (what ==  PyTrace_C_CALL || what == PyTrace_C_RETURN) {
      PyCFunctionObject *fn = (PyCFunctionObject*)args;
      PyObject *name = PyUnicode_FromString(fn->m_ml->ml_name);
      //func_name = repr(name);
      PyObject *r = PyObject_Repr(name);
      PyObject *s = PyUnicode_AsEncodedString(r, "utf-8", "~E~");
      func_name = strdup(PyBytes_AS_STRING(s));
      Py_XDECREF(r);
      Py_XDECREF(s);

      //func_name = (char*)malloc((strlen(func_name) + 3) * sizeof(char));
      //strcpy (func_name, buf);
      //strcat (func_name, "_C");
      //printf ("FUNC: %s\n", func_name);

      pseudo_addr = vftr_jenkins_murmur_64_hash (strlen(func_name), (uint8_t*)func_name);
      pseudo_addr += (uint64_t)frame->f_back;
      //free(buf);
      printf ("%s: %s @ 0x%lx\n", what == PyTrace_C_CALL ? "C Entry": "C Exit", func_name, pseudo_addr);
   } else if (what == PyTrace_EXCEPTION) {
      //printf ("exception\n");
   } else if (what == PyTrace_C_EXCEPTION) {
      //printf ("C exception\n");
   } else {
      printf ("UNKNOWN!\n");
   }
   //return 0;
   //printf ("func_name: %s\n", func_name);
   //if (!func_name) return 0;
   
   if (what == PyTrace_CALL || what == PyTrace_C_CALL) {
   //if (what == PyTrace_CALL) {
      thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
      threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);

      //vftr_stack_t *my_stack = vftrace.process.stacktree.stacks + my_threadstack->stackID;
      //if (what == PyTrace_CALL && !strcmp(func_name, "'acquire'") && !strcmp(my_stack->name, "'flush'")) {
      //   printf ("Caller: %s(%d)\n", vftr_get_stack_string(vftrace.process.stacktree, my_threadstack->stackID, false), my_threadstack->stackID);
      //}
      //profile_t *my_profile = vftr_get_my_profile(my_stack, my_thread);

      my_threadstack = vftr_update_threadstack_region (my_threadstack, my_thread,
                                                       (uintptr_t)pseudo_addr, func_name,
                                                       &vftrace, false);
      vftr_stack_t *my_stack = vftrace.process.stacktree.stacks + my_threadstack->stackID;
      if (what == PyTrace_CALL && !strcmp(func_name, "'acquire'")) printf ("New stack: %d\n", my_threadstack->stackID);
      profile_t *my_profile = vftr_get_my_profile(my_stack, my_thread);
      vftr_accumulate_callprofiling(&(my_profile->callprof), 1, -function_time_begin);
   } else if (what == PyTrace_RETURN || what == PyTrace_C_RETURN) {
   //} else if (what == PyTrace_RETURN) {
      if (!strcmp (func_name, "'<module>'")) return 0;
      thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
      threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);
      vftr_stack_t *my_stack = vftrace.process.stacktree.stacks + my_threadstack->stackID;
      profile_t *my_profile = vftr_get_my_profile(my_stack, my_thread);
      vftr_accumulate_callprofiling(&(my_profile->callprof), 0, function_time_begin);
      (void)vftr_threadstack_pop(&(my_thread->stacklist));
   }
   return 0;
}

static PyObject *init_vftrace (PyObject *self) {
  PyEval_SetProfile(profiler_callback, self);
  PyFrameObject *f = PyEval_GetFrame();
  while (f != NULL) {
     PyCodeObject *fn = (PyCFunctionObject*)f->f_code;
     PyObject *r = PyObject_Repr(fn->co_name);
     PyObject *s = PyUnicode_AsEncodedString(r, "utf-8", "~E~");
     char *func_name = strdup(PyBytes_AS_STRING(s));
     Py_XDECREF(r);
     Py_XDECREF(s);
     printf ("Resolve Frame %s\n", func_name);
     f = f->f_back;
  }
  vftr_initialize (NULL, NULL);
  Py_RETURN_NONE;
}

static struct PyModuleDef vftrace_definition = {
   PyModuleDef_HEAD_INIT,
   .m_name = "vftrace",
   .m_doc = "Python interface for Vftrace",
   //.m_size = ???
   .m_methods = vftraceMethods,
   //.m_clear = vftrace_clear_m,
   //.m_free = vftrace_free_m,
   //.m_slots = ???
   //.m_traverse = ???
};

PyMODINIT_FUNC PyInit_vftrace(void) {
   Py_Initialize();
   PyObject *thisPy = PyModule_Create(&vftrace_definition);
   //PyModule_AddType(thisPy, &
   return thisPy;
}

