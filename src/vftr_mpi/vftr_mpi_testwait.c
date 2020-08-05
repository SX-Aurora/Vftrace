/*
   This file is part of Vftrace.

   Vftrace is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.

   Vftrace is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License along
   with this program; if not, write to the Free Software Foundation, Inc.,
   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

#include <stdlib.h>

#ifdef _MPI
#include <mpi.h>

#include "vftr_mpi_environment.h"
#include "vftr_async_messages.h"
  
int vftr_MPI_Barrier(MPI_Comm comm) {
   // there is no timing needed but we keep the wrapper here
   // for the instrumentation
   return PMPI_Barrier(comm);
}

int vftr_MPI_Test(MPI_Request *request, int *flag, MPI_Status *status) {

   // disable profiling based on the Pcontrol level
   if (vftrace_Pcontrol_level == 0) {
      return PMPI_Test(request, flag, status);
   } else {
      // do not call MPI_Test immediately!
      // If the communication is successfull MPI_Test destroys the Reqeust 
      // Hence, no chance of properly clearing the communication
      // from the open request list
      // MPI_Request_get_status is a non destructive check. 
      int retVal = PMPI_Request_get_status(*request, flag, status);
   
      if (*flag) {
         // Communication is done.
         // Clear finished communications from the open request list
         vftr_clear_completed_request();
         // Now that the danger of deleating needed requests is banned
         // actually call MPI_Test   
         retVal = PMPI_Test(request, flag, status);
      }
   
      return retVal;
   }
}

int vftr_MPI_Testany(int count, MPI_Request array_of_requests[],
                     int *index, int *flag, MPI_Status *status) {

   // disable profiling based on the Pcontrol level
   if (vftrace_Pcontrol_level == 0) {
      return PMPI_Testany(count, array_of_requests, index, flag, status);
   } else {
      // First check if the request array contains at least one active handle
      bool activereqs = false;
      for (int ireq=0; ireq<count; ireq++) {
         if (vftr_mpi_request_is_active(array_of_requests[ireq])) {
            activereqs = true;
            break;
         }
      }
      // if no active request is found return with the following settings
      if (!activereqs) {
         *flag = true;
         *index = MPI_UNDEFINED;
         if (status != MPI_STATUS_IGNORE) {
            vftr_empty_mpi_status(status);
         }
         return MPI_SUCCESS;
      }

      // initialize the index to the default failure value
      *index = MPI_UNDEFINED;
      // loop over all requests and terminate the loop
      // on the first completed communication
      int retVal = 0;
      for (int ireq=0; ireq<count; ireq++) {
         retVal = PMPI_Request_get_status(array_of_requests[ireq],
                                          flag,
                                          status);
         // terminate if a completed communication is found
         if (flag) {
            // record the index of the completed communication
            *index = ireq;
            // clear completed communication from the list of open requests
            vftr_clear_completed_request();
            // Mark the request as inactive, or deallocate it.
            retVal = PMPI_Test(array_of_requests+ireq,
                               flag,
                               status);
            break;
         }
      }

      return retVal;
   }
}

int vftr_MPI_Testsome(int incount, MPI_Request array_of_requests[],
                      int *outcount, int array_of_indices[],
                      MPI_Status array_of_statuses[]) {

   // disable profiling based on the Pcontrol level
   if (vftrace_Pcontrol_level == 0) {
      return PMPI_Testsome(incount, array_of_requests, outcount,
                           array_of_indices, array_of_statuses);
   } else {
      // First check if the request array contains at least one active handle
      bool activereqs = false;
      for (int ireq=0; ireq<incount; ireq++) {
         if (vftr_mpi_request_is_active(array_of_requests[ireq])) {
            activereqs = true;
            break;
         }
      }
      // if no active request is found return with the following settings
      if (!activereqs) {
         *outcount = MPI_UNDEFINED;
         return MPI_SUCCESS;
      }

      int retVal = MPI_SUCCESS;
      int tmpretVal;
      // loop over all requests and check for completion
      *outcount = 0;
      for (int ireq=0; ireq<incount; ireq++) {
         int flag;
         if (array_of_statuses == MPI_STATUSES_IGNORE) {
            tmpretVal = PMPI_Request_get_status(array_of_requests[ireq],
                                                &flag,
                                                MPI_STATUS_IGNORE);
         } else {
            tmpretVal = PMPI_Request_get_status(array_of_requests[ireq],
                                                &flag,
                                                array_of_statuses+ireq);
         }
         if (tmpretVal != MPI_SUCCESS) {
            // if something goes wrong inform the
            // user to check the status variable
            retVal = MPI_ERR_IN_STATUS;
         }
         if (flag) {
            // record completed communications for return
            (*outcount)++;
            array_of_indices[(*outcount)-1] = ireq;
            // remove completed communications from the list of open requests
            vftr_clear_completed_request();
            // Mark the request as inactive, or deallocate it.
            if (array_of_statuses == MPI_STATUSES_IGNORE) {
               tmpretVal = PMPI_Test(array_of_requests+ireq,
                                     &flag,
                                     MPI_STATUS_IGNORE);
            } else {
               tmpretVal = PMPI_Test(array_of_requests+ireq,
                                     &flag,
                                     array_of_statuses+ireq);
            }
            if (tmpretVal != MPI_SUCCESS) {
               // if something goes wrong inform the
               // user to check the status variable
               retVal = MPI_ERR_IN_STATUS;
            }
         }
      }

      return retVal;
   }
}

int vftr_MPI_Testall(int count, MPI_Request array_of_requests[],
                     int *flag, MPI_Status array_of_statuses[]) {

   // disable profiling based on the Pcontrol level
   if (vftrace_Pcontrol_level == 0) {
      return PMPI_Testall(count, array_of_requests, flag, array_of_statuses);
   } else {
      int retVal;
      int tmpflag;

      // set the return flag to true
      *flag = true;
      // It will be returned true if all communications are completed
      for (int ireq=0; ireq<count; ireq++) {
         if (array_of_statuses == MPI_STATUSES_IGNORE) {
            retVal = PMPI_Request_get_status(array_of_requests[ireq],
                                             &tmpflag,
                                             MPI_STATUS_IGNORE);
         } else {
            retVal = PMPI_Request_get_status(array_of_requests[ireq],
                                             &tmpflag,
                                             array_of_statuses+ireq);
         }
         // accumulate truthness of the individual requests
         // as soon as one if false everything is
         *flag = (*flag) && tmpflag;
      }
      // clear completed communications from the list of open requests
      vftr_clear_completed_request();

      if (flag) {
      // If all communications are completed
      // run Testall to modify the requests appropriately
         retVal = PMPI_Testall(count, array_of_requests, flag, array_of_statuses);
      }

      return retVal;
   }
}

int vftr_MPI_Wait(MPI_Request *request, MPI_Status *status) {

   // disable profiling based on the Pcontrol level
   if (vftrace_Pcontrol_level == 0) {
      return PMPI_Wait(request, status);
   } else {
      int retVal;

      // loop until the communication corresponding to the request is completed
      int flag = false;
      while (!flag) {
         // check if the communication is finished
         retVal = PMPI_Request_get_status(*request,
                                          &flag,
                                          status);
         // either the communication is completed, or not
         // other communications might be completed in the background
         // clear those from the list of open requests
         vftr_clear_completed_request();
      }
      // Properly set the request and status variable
      retVal = PMPI_Wait(request, status);

      return retVal;
   }
}

int vftr_MPI_Waitany(int count, MPI_Request array_of_requests[],
                     int *index, MPI_Status *status) {

   // disable profiling based on the Pcontrol level
   if (vftrace_Pcontrol_level == 0) {
      return PMPI_Waitany(count, array_of_requests, index, status);
   } else {
      if (count <= 0) {return MPI_SUCCESS;}
   
      // First check if the request array contains at least one active handle
      bool activereqs = false;
      for (int ireq=0; ireq<count; ireq++) {
         if (vftr_mpi_request_is_active(array_of_requests[ireq])) {
            activereqs = true;
            break;
         }
      }
      // if no active request is found return with the following settings
      if (!activereqs) {
         *index = MPI_UNDEFINED;
         if (status == MPI_STATUS_IGNORE) {
            vftr_empty_mpi_status(status);
         }
         return MPI_SUCCESS;
      }
   
      // loop until at least one communication of the requests is completed
      int retVal;
      int completed_req = false;
      while (!completed_req) {
         // loop over all requests
         for (int ireq=0; ireq<count; ireq++) {
            int flag;
            // check if the communication associated with the request
            // is completed
            retVal = PMPI_Request_get_status(array_of_requests[ireq],
                                             &flag,
                                             status);
            completed_req = completed_req || flag;
            // either the communication is completed, or not
            // other communications might be completed in the background
            // clear those from the list of open requests
            vftr_clear_completed_request();
            // if this request corresponds to a completed communication
            // leave the loop
            if (flag) {
               // record the index of the finished request
               *index = ireq;
               break;
            }
         }
      }
   
      // Properly set the request and status variable
      retVal = PMPI_Wait(array_of_requests+(*index), status);
   
      return retVal;
   }
}

int vftr_MPI_Waitsome(int incount, MPI_Request array_of_requests[],
                      int *outcount, int array_of_indices[],
                      MPI_Status array_of_statuses[]) {

   // disable profiling based on the Pcontrol level
   if (vftrace_Pcontrol_level == 0) {
      return PMPI_Waitsome(incount, array_of_requests, outcount,
                           array_of_indices, array_of_statuses);
   } else {
      if (incount <= 0) {
         outcount = 0;
         return MPI_SUCCESS;
      }

      // First check if the request array contains at least one active handle
      bool activereqs = false;
      for (int ireq=0; ireq<incount; ireq++) {
         if (vftr_mpi_request_is_active(array_of_requests[ireq])) {
            activereqs = true;
            break;
         }
      }
      // if no active request is found return with the following settings
      if (!activereqs) {
         *outcount = MPI_UNDEFINED;
         return MPI_SUCCESS;
      }

      int retVal = MPI_SUCCESS;
      int tmpretVal;
      *outcount = 0;
      // loop while outcount is 0
      while (*outcount == 0) {
         // loop over all requests and check for completion
         for (int ireq=0; ireq<incount; ireq++) {
            int flag;
            if (array_of_statuses == MPI_STATUSES_IGNORE) {
               tmpretVal = PMPI_Request_get_status(array_of_requests[ireq],
                                                   &flag,
                                                   MPI_STATUS_IGNORE);
            } else {
               tmpretVal = PMPI_Request_get_status(array_of_requests[ireq],
                                                   &flag,
                                                   array_of_statuses+ireq);
            }
            if (tmpretVal != MPI_SUCCESS) {
               // if something goes wrong inform the
               // user to check the status variable
               retVal = MPI_ERR_IN_STATUS;
            }
            if (flag) {
               // record completed communications for return
               (*outcount)++;
               array_of_indices[(*outcount)-1] = ireq;
               // remove completed communications from the list of open requests
               vftr_clear_completed_request();
               // Mark the request as inactive, or deallocate it.
               if (array_of_statuses == MPI_STATUSES_IGNORE) {
                  tmpretVal = PMPI_Wait(array_of_requests+ireq,
                                        MPI_STATUS_IGNORE);
               } else {
                  tmpretVal = PMPI_Wait(array_of_requests+ireq,
                                        array_of_statuses+ireq);
               }
               if (tmpretVal != MPI_SUCCESS) {
                  // if something goes wrong inform the
                  // user to check the status variable
                  retVal = MPI_ERR_IN_STATUS;
               }
            }
         }
      }

      return retVal;
   }
}

int vftr_MPI_Waitall(int count, MPI_Request array_of_requests[],
                     MPI_Status array_of_statuses[]) {

   // disable profiling based on the Pcontrol level
   if (vftrace_Pcontrol_level == 0) {
      return PMPI_Waitall(count, array_of_requests, array_of_statuses);
   } else {
      if (count <= 0) {return MPI_SUCCESS;}
   
      // loop while at least one request is not completed
      int *req_completed = (int*) malloc(count*sizeof(int));
      for (int ireq=0; ireq<count; ireq++) {
         req_completed[ireq] = false;
      }
      int retVal;
      bool all_completed = false;
      while (!all_completed) {
         all_completed = true;
         // loop over all requests
         for (int ireq=0; ireq<count; ireq++) {
            if (!req_completed[ireq]) {
               // check if the communication associated with the request
               // is completed
               PMPI_Request_get_status(array_of_requests[ireq],
                                       req_completed+ireq,
                                       MPI_STATUS_IGNORE);
               // if not completed 
               if (!(req_completed[ireq])) {
                  all_completed = false;
               }
            }
         }
         vftr_clear_completed_request();
      }

      free(req_completed);
      req_completed = NULL;
   
      return PMPI_Waitall(count, array_of_requests, array_of_statuses);
   }
}

#endif
