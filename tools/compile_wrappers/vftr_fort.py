#!/usr/bin/env python

import sys
import re

filename_in = sys.argv[1]
filename_out = filename_in + ".vftr"
allocate_pattern = re.compile("^[ ]*allocate[\ ,\(]", re.IGNORECASE)
deallocate_pattern = re.compile("^[ ]*deallocate[\ ,\(]", re.IGNORECASE)
subroutine_pattern = re.compile("^[ ]*subroutine[\ ,\(]", re.IGNORECASE)
function_pattern = re.compile("^[ ]*function[\ ,\(]", re.IGNORECASE)

def split_alloc_argument (arg, check_any_bracket=False):
  open_bracket = False
  any_full_bracket = not check_any_bracket
  all_args = []
  tmp = ""
  for char in arg:
    if not open_bracket and any_full_bracket and char == ",":
      all_args.append(tmp)
      tmp = ""
      any_full_bracket = not check_any_bracket
    else: 
      if char == "(":
        open_bracket = True
      elif char == ")":
        if open_bracket: any_full_bracket = True
        open_bracket = False
      tmp += char
  if any_full_bracket: all_args.append (tmp)
  return all_args

def split_line (tot_string, is_alloc, is_dealloc):
   # Remove white spaces and ampersands (&)
   tot_string = re.sub(r"\s+", "", tot_string) # This also removes intermediate newlines, for some reason
   tot_string = re.sub(r"&+", "", tot_string)
   # Get everything in between the outer brackets (...)
   tot_string = tot_string[tot_string.find("(")+1:-1]
   # Get all the fields
   #print ("Before splitting: ", tot_string)
   fields = split_alloc_argument (tot_string, check_any_bracket=True)

   #if is_alloc:
   #  fields = tot_string.split(",")
   #elif is_dealloc:
   #  fields = tot_string.split(",")
   #print ("After splitting: ", fields)
   # Check if there is a "STAT" argument in the last element of the list. If so, we return the list reduced by one element.
   #if re.search ("STAT", fields[-1], re.IGNORECASE):
   #  fields.pop()
   return fields

def construct_vftrace_allocate_call (field):
  # Input: name(n1,n2,... 
  # We obtain the field name by taking everything before the first opening bracket.
  # The array dimensions can be given by functions, e.g. ALLOCATE (arr(f(x))). 
  # Therefore, it is important to only match the first bracket, not proceeding ones, which would be
  # done with a call to "split". 
  bracket_index = field.find("(") 
  #print ("FIELD:", field)
  name = field[0:bracket_index]
  rest = field[bracket_index+1:-1]
  #print ("NAME: ", name, "REST: ", rest)
  # Split at the commas. The first element is "<name>(<first_dim>", so we split again at the bracket.
  dims = split_alloc_argument (rest)
  #print ("dims: ", dims)
  # Create the string which computes the total number of elements in the field by multiplying the individual dimensions.
  dim_string = ""
  for i, dim in enumerate(dims):
    # If there is a colon ("x1:x2") in the string, the dimension size is x2 - x1 + 1.
    # Otherwise, the dimension bounds start at 1, and the size is dim.
    if ":" in dim:
      tmp = dim.split(":")
      #print ("tmp: ", tmp)
      dim_string += "(" + tmp[1] + "-" + tmp[0] + "+1)"
    else:
      dim_string += dim
    if i + 1 < len(dims):
      dim_string += "*"
  return "call vftrace_allocate(\"" + name + "\", " + dim_string + ", storage_size(" + name + ")/8)\n"

def construct_vftrace_deallocate_call (field):
  # The input is simply the field name.
  return "call vftrace_deallocate(\"" + field + "\")\n"

def line_to_be_continued(line):
  line_wo_spaces = re.sub(r"\s+", "", line)  
  return line_wo_spaces[-1] == "&"


with open(filename_in, "r") as f_in, open(filename_out, "w") as f_out:
  all_lines = f_in.readlines()
  n_lines = len(all_lines)
  subroutine_start = False
  subroutine_end = False
  skip_subroutine = False
  for i_line, line in enumerate(all_lines):
    is_alloc = allocate_pattern.match(line)
    is_dealloc = deallocate_pattern.match(line)
    # Put "use vftrace" after every subroutine or function definition, regardless if it is actually used.
    is_subroutine = subroutine_pattern.match(line)
    is_function = function_pattern.match(line)

    # We need to find out when the function definition has been written completely in the previous iteration.
    # On flag indicates that the subroutine definition has been started. If it is set, we check if it is finished.
    # If the latter one is set before all the other ones, it's time to insert the use statement.
    if subroutine_end and not skip_subroutine:
       f_out.write ("use vftrace\n")
       subroutine_end = False
    if is_subroutine or is_function:
       if re.search("pure ", line, re.IGNORECASE) or re.search("elemental ", line, re.IGNORECASE):
         skip_subroutine = True
       else:
         skip_subroutine = False
         subroutine_start = True
    if subroutine_start:
       tmp = re.sub(r"\s+", "", line)
       if (tmp[-1] != "&"):
         subroutine_end = True
         subroutine_start = False

    # Register allocate and deallocate calls.
    if not skip_subroutine and (is_alloc or is_dealloc):
      tot_string = line
      # Concatenate line breaks indicated by ampersands "&"
      line_tmp = line
      i = i_line
      while line_to_be_continued(line_tmp):
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
