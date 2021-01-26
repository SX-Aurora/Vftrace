!*******************************************************************************
!                                                                              *
!   This file is part of vftrace.                                              *
!                                                                              *
!   Vftrace is free software; you can redistribute it and/or modify            *
!   it under the terms of the GNU General Public License as published by       *
!   the Free Software Foundation; either version 2 of the License, or          *
!   (at your option) any later version.                                        *
!                                                                              *
!   Vftrace is distributed in the hope that it will be useful,                 *
!   but WITHOUT ANY WARRANTY; without even the implied warranty of             *
!   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
!   GNU General Public License for more details.                               *
!                                                                              *
!   You should have received a copy of the GNU General Public License along    *
!   with this program; if not, write to the Free Software Foundation, Inc.,    *
!   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.                *
!                                                                              *
!*******************************************************************************

!===============================================================================
!
!  User Fortran interface to vftrace
!
!===============================================================================

#define import(x) import x

module vftrace

   implicit none
   
   private

   public :: vftrace_region_begin, &
             vftrace_region_end
   public :: vftrace_allocate, &
             vftrace_deallocate
   public :: vftrace_get_stack
   public :: vftrace_pause, &
             vftrace_resume

   public :: vftrace_get_stacktree_size

   interface

      ! pause and resume sampling via vftrace in user code
      subroutine vftrace_pause() &
         bind(c, name="vftrace_pause")
         implicit none
      end subroutine
 
      subroutine vftrace_resume() &
         bind(c, name="vftrace_resume")
         implicit none
      end subroutine

   end interface

   interface 

      function vftrace_get_stacktree_size() &
         bind(c, name="vftrace_get_stacktree_size")
         use iso_c_binding, only: c_int
         implicit none
         integer(kind=c_int) :: vftrace_get_stacktree_size
      end function vftrace_get_stacktree_size

   end interface

   ! interface for non public regions routines
   interface
      ! Mark the entry of an instrumented region
      ! name ist the Region name to be used in the profile
      subroutine vftrace_region_begin_C(name) &
         bind(c, name="vftrace_region_begin")
         use iso_c_binding, only : c_char
         implicit none
         character(c_char), intent(in) :: name(*)
      end subroutine vftrace_region_begin_C
 
      ! Mark end of instrumented region
      ! name ist the Region name to be used in the profile
      subroutine vftrace_region_end_C(name) &
         bind(c, name="vftrace_region_end")
         use iso_c_binding, only : c_char
         implicit none
         character(c_char), intent(in) :: name(*)
      end subroutine vftrace_region_end_C

      !!!subroutine vftrace_allocate_C (name, dims, n_dims, element_size) &
      subroutine vftrace_allocate_C (name, n_elements, element_size) &
         bind(c, name="vftrace_allocate")
         use iso_c_binding, only: c_char, c_int
         implicit none
         character(c_char), intent(in) :: name(*)
         !!!integer(c_int), intent(in) :: dims(*)
         integer(c_int), intent(in) :: n_elements
         !!!integer(c_int), intent(in) :: n_dims
         integer(c_int), intent(in) :: element_size
      end subroutine vftrace_allocate_C
  
      subroutine vftrace_deallocate_C (name) bind(c, name="vftrace_deallocate")
         use iso_c_binding, only: c_char
         implicit none
         character(c_char), intent(in) :: name(*)
      end subroutine
   end interface

   ! interface for non public stack routines
   ! used for passing string pointer between fortran and C
   interface
      function vftrace_get_stack_C_charptr() &
         bind(C, name="vftrace_get_stack")
         use iso_c_binding, only : c_ptr
         implicit none
         type(c_ptr) :: vftrace_get_stack_C_charptr
      end function vftrace_get_stack_C_charptr

      function vftrace_get_stack_string_length_C_int() &
         bind(C, name="vftrace_get_stack_string_length")
         use iso_c_binding, only : c_int
         implicit none
         integer(kind=c_int) :: vftrace_get_stack_string_length_C_int
      end function vftrace_get_stack_string_length_C_int

   end interface

contains

   subroutine vftrace_region_begin(name)
      use iso_c_binding, only : c_char, c_null_char
      implicit none
      character(len=*), intent(in) :: name
      integer :: name_len
      character(kind=c_char,len=:), allocatable :: c_name
      name_len = len(adjustl(trim(name)))
      ! null terminator space 
      name_len = name_len + 1
      allocate(character(len=name_len) :: c_name)
      c_name(:) = adjustl(trim(name))
      c_name(name_len:name_len+1) = c_null_char
      call vftrace_region_begin_C(c_name)
      deallocate(c_name)
   end subroutine vftrace_region_begin

   subroutine vftrace_region_end(name)
      use iso_c_binding, only : c_char, c_null_char
      implicit none
      character(len=*), intent(in) :: name
      integer :: name_len
      character(kind=c_char,len=:), allocatable :: c_name
      name_len = len(adjustl(trim(name)))
      ! null terminator space 
      name_len = name_len + 1
      allocate(character(len=name_len) :: c_name)
      c_name(:) = adjustl(trim(name))
      c_name(name_len:name_len+1) = c_null_char
      call vftrace_region_end_C(c_name)
      deallocate(c_name)
   end subroutine vftrace_region_end

   function vftrace_get_stack()
      use iso_c_binding, only : c_int, &
                                c_f_pointer
      implicit none

      integer(kind=c_int) :: stringlength
      character(len=:), pointer :: vftrace_get_stack
      character, pointer, dimension(:) :: tmpstring

      integer :: i

      stringlength = vftrace_get_stack_string_length_C_int()
      CALL C_F_POINTER(vftrace_get_stack_C_charptr(), tmpstring, [stringlength])

      ALLOCATE(character(len=stringlength) :: vftrace_get_stack)
      do i = 1, stringlength
         vftrace_get_stack(i:i+1) = tmpstring(i)
      end do
   end function vftrace_get_stack

   !!!subroutine vftrace_allocate (name, n1)
   subroutine vftrace_allocate (name, n_elements, element_size)
     use iso_c_binding, only: c_char, c_null_char, c_int
     implicit none
     character(len=*), intent(in) :: name
     !!!integer, intent(in), dimension(:) :: n1
     integer, intent(in) :: n_elements
     integer, intent(in) :: element_size
     integer :: name_len, index_len
     character(kind=c_char,len=:), allocatable :: c_name
     !!!integer(kind=c_int), dimension(:), allocatable :: n1_C
     !!!integer :: i, element_size
     integer :: i
     name_len = len(adjustl(trim(name))) + 1
     allocate (character(len=name_len) :: c_name)
     c_name(:) = adjustl(trim(name))
     c_name(name_len:name_len+1) = c_null_char
     !!!index_len = size(n1)
     !!!print *, 'n1: ', n1, size(n1)
     !!!element_size = storage_size(n1) / 8 
     !!!allocate (n1_C (index_len))
     !!!do i = 1, index_len
     !!!  n1_C(i) = int(n1(i), c_int)
     !!!end do
     !!!n1_C = int(n1, c_int) 
     !!!print *, 'location: ', loc(n1_C)
     !!!call vftrace_allocate_C (c_name, n1_C, index_len, element_size)
     call vftrace_allocate_C (c_name, int(n_elements, c_int), int(element_size, c_int))
     !!!call vftrace_allocate_1_C (c_name)
     deallocate (c_name)
     !!!deallocate (n1_C)
   end subroutine vftrace_allocate

   subroutine vftrace_deallocate (name)
     use iso_c_binding, only: c_char, c_null_char
     implicit none
     character(len=*), intent(in) :: name
     integer :: name_len
     character(kind=c_char,len=:), allocatable :: c_name
     name_len = len(adjustl(trim(name))) + 1
     allocate (character(len=name_len) :: c_name)
     c_name(:) = adjustl(trim(name))
     c_name(name_len:name_len+1) = c_null_char
     call vftrace_deallocate_C (c_name)
     deallocate (c_name)
   end subroutine
end module vftrace
