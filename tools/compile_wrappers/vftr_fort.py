#!/usr/bin/env python

import sys
import re

filename_in = sys.argv[1]
filename_out = filename_in + ".vftr"
allocate_pattern = re.compile("^[ ]*allocate[\ ,\(]", re.IGNORECASE)
deallocate_pattern = re.compile("^[ ]*deallocate[\ ,\(]", re.IGNORECASE)
subroutine_pattern = re.compile("^[ ]*subroutine[\ ,\(]", re.IGNORECASE)
function_pattern = re.compile("^[ ]*function[\ ,\(]", re.IGNORECASE)

def split_line (tot_string, is_alloc, is_dealloc):
   # Remove white spaces and ampersands (&)
   tot_string = re.sub(r"\s+", "", tot_string) # This also removes intermediate newlines, for some reason
   tot_string = re.sub(r"&+", "", tot_string)
   # Get everything in between the outer brackets (...)
   tot_string = tot_string[tot_string.find("(")+1:-2]
   # Get all the fields
   if is_alloc:
     fields = tot_string.split("),")
   elif is_dealloc:
     fields = tot_string.split(",")
   # Check if there is a "STAT" argument in the last element of the list. If so, we return the list reduced by one element.
   if re.search ("STAT", fields[-1], re.IGNORECASE):
     fields.pop()
   return fields

def construct_vftrace_allocate_call (field):
  # Input: name(n1,n2,... 
  # Split at the commas. The first element is "<name>(<first_dim>", so we split again at the bracket.
  tmp = field.split(",")
  tmp2 = tmp[0].split("(")
  name = tmp2[0]
  dims = [tmp2[1]] + tmp[1:]
  dim_string = ""
  for i, dim in enumerate(dims):
    # If there is a colon ("x1:x2") in the string, the dimension size is x2 - x1 + 1.
    #Otherwise, the dimension bounds start at 1, and the size is dim.
    if ":" in dim:
      tmp = dim.split(":")
      dim_string += "(" + tmp[1] + "-" + tmp[0] + "+1)"
    else:
      dim_string += dim
    if i + 1 < len(dims):
      dim_string += "*"
  return "call vftrace_allocate(" + name + ", " + dim_string + ", storage_size(" + name + ")/8)\n"

def construct_vftrace_deallocate_call (field):
  # The input is simply the field name.
  return "call vftrace_deallocate(" + field + ")\n"


with open(filename_in, "r") as f_in, open(filename_out, "w") as f_out:
  all_lines = f_in.readlines()
  n_lines = len(all_lines)
  subroutine_start = False
  subroutine_end = False
  for i_line, line in enumerate(all_lines):
    is_alloc = allocate_pattern.match(line)
    is_dealloc = deallocate_pattern.match(line)
    # Put "use vftrace" after every subroutine or function definition, regardless if it is actually used.
    is_subroutine = subroutine_pattern.match(line)
    is_function = function_pattern.match(line)
    # We need to find out, when the function definition has been written completely in the previous iteration.
    # On flag indicates that the subroutine definition has been started. If it is set, we check if it is finished.
    # If the latter one is set before all the other ones, it's time to insert the use statement.
    if subroutine_end:
       f_out.write ("use vftrace")
       subroutine_end = False
    if is_subroutine or is_function:
       print ("LINE: ", line)
       subroutine_start = True
    if subroutine_start:
       tmp = re.sub(r"\s+", "", line)
       print ("tmp: ", tmp)
       if (tmp[-1] != "&"):
         subroutine_end = True
         subroutine_start = False
    if is_alloc or is_dealloc:
      # Concatenate line breaks indicated by ampersands "&"
      tot_string = line
      line_tmp = line
      i = i_line
      while "&" in line_tmp: 
        i = i + 1
        if i < n_lines:
          line_tmp = all_lines[i]
        else:
          break
        tot_string += line_tmp
      fields = split_line(tot_string, is_alloc, is_dealloc)
      for field in fields:
        if is_alloc:
          f_out.write (construct_vftrace_allocate_call(field))
        elif is_dealloc:
          f_out.write (construct_vftrace_deallocate_call(field))

    f_out.write (line)
