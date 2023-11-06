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
#include "stacks.h"


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
      PyObject *r = PyObject_Repr(fn->co_name);
      PyObject *s = PyUnicode_AsEncodedString(r, "utf-8", "~E~");
      func_name = strdup(PyBytes_AS_STRING(s));
      Py_XDECREF(r);
      Py_XDECREF(s);
      pseudo_addr = vftr_jenkins_murmur_64_hash (strlen(func_name), (uint8_t*)func_name);
   } else if (what ==  PyTrace_C_CALL || what == PyTrace_C_RETURN) {
      PyCFunctionObject *fn = (PyCFunctionObject*)args;
      PyObject *name = PyUnicode_FromString(fn->m_ml->ml_name);
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
   } else if (what == PyTrace_EXCEPTION) {
      //printf ("exception\n");
   } else if (what == PyTrace_C_EXCEPTION) {
      //printf ("C exception\n");
   } else {
      printf ("UNKNOWN!\n");
   }
   
   if (what == PyTrace_CALL || what == PyTrace_C_CALL) {
      thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
      threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);

      my_threadstack = vftr_update_threadstack_region (my_threadstack, my_thread,
                                                       (uintptr_t)pseudo_addr, func_name,
                                                       &vftrace, false);
      vftr_stack_t *my_stack = vftrace.process.stacktree.stacks + my_threadstack->stackID;
      profile_t *my_profile = vftr_get_my_profile(my_stack, my_thread);
      vftr_accumulate_callprofiling(&(my_profile->callprof), 1, -function_time_begin);
   } else if (what == PyTrace_RETURN || what == PyTrace_C_RETURN) {
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
  vftr_initialize (NULL, NULL);
  stacktree_t *vftr_stacktree = &(vftrace.process.stacktree);
  if (!(vftr_stacktree->nstacks) > 1) {
     printf ("Internal Vftrace error: Python tracing starts with existing stacktree!\n");
  } else {
     printf ("Stacks in tree: %d\n", vftr_stacktree->nstacks);
  }

  // Inject current stack  
  // We need to invert the order of the Python frames. Therefore, we first determine the stack depth.
  int py_stack_size = 0;
  PyFrameObject *f = PyEval_GetFrame();
  while (f != NULL) {
     py_stack_size++;
     f = f->f_back;
  }
  PyFrameObject **fns = (PyFrameObject**)malloc(py_stack_size * sizeof(PyFrameObject*));
  f = PyEval_GetFrame();
  int i = 0;
  while (f != NULL) {
     fns[i++] = f;
     f = f->f_back;
  }
  for (int i_stack = py_stack_size - 1; i_stack >= 0; i_stack--) {
     PyCodeObject *fn = (PyCFunctionObject*)fns[i_stack]->f_code;
     PyObject *r = PyObject_Repr(fn->co_name);
     PyObject *s = PyUnicode_AsEncodedString(r, "utf-8", "~E~");
     char *func_name = strdup(PyBytes_AS_STRING(s));
     Py_XDECREF(r);
     Py_XDECREF(s);
     int calleeID = vftr_new_stack (py_stack_size - i_stack - 1, vftr_stacktree,
                                    func_name, func_name, (uintptr_t)f, false); 
     thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
     vftr_threadstack_push (calleeID, &(my_thread->stacklist));
  }

  // Register the callback
  PyEval_SetProfile(profiler_callback, self);
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

