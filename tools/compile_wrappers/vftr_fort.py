#!/usr/bin/env python

import sys
import re

filename_in = sys.argv[1]
filename_out = filename_in + ".vftr"
allocate_pattern = re.compile("^[ ]*allocate[\ ,\(]", re.IGNORECASE)
deallocate_pattern = re.compile("^[ ]*deallocate[\ ,\(]", re.IGNORECASE)
#subroutine_pattern = re.compile(r"^[\S\s]*(pure)?(elemental)?[\S\s]*subroutine[\s]+[\S]*[ ]*(\()?", re.IGNORECASE)
subroutine_pattern = re.compile(r"^[\S\s]*(pure[\s]+)?(elemental[\s]+)?subroutine[\s]+[\S]*[ ]*(\()?", re.IGNORECASE)
#function_pattern = re.compile(r"^[\S\s]*(pure)?(elemental)?[\S\s]*function[\s]+[\S]*[ ]*(\()?", re.IGNORECASE)
function_pattern = re.compile(r"^[\S\s]*(pure[\s]+)?(elemental[\s]+)?function[\s]+[\S]*[ ]*(\()?", re.IGNORECASE)
end_routine_pattern = re.compile(r"[ ]*end[ ]*(function|subroutine)", re.IGNORECASE)

vftrace_wrapper_marker = "!!! VFTRACE 1.3 - Wrapper inserted malloc-trace calls\n"

def check_if_function_or_subroutine (line):
  if subroutine_pattern.match(line) or function_pattern.match(line):
    ll = line.lower()
    if end_routine_pattern.match(ll):
      return False
    pos1 = ll.find("!")
    pos2 = ll.find("function")
    pos3 = ll.find("subroutine")
    if (pos1 >= 0): # There is a comment. Find out if it is behind the function definition.
      value = (pos2 >= 0 and pos2 < pos1) or (pos3 >= 0 and pos3 < pos1)
    else:
      value = True
  else:
    value = False
  # Check if the pattern appears as part of a string.
  if value:
    pos4 = [i for i, this_char in enumerate(line) if this_char == "\""]
    pos5 = [i for i, this_char in enumerate(line) if this_char == "\'"]
    # If it is a proper routine definition, there can be no " before the position of the patterns (pos2, pos3).
    # We therefore only check if there is any pos4 or pos5 smaller than that.
    if pos2 >= 0 and pos3 >= 0:
      pos = pos2 if pos2 < pos3 else pos3
    elif pos2 >= 0:
      pos = pos2
    elif pos3 >= 0:
      pos = pos3
    for p in pos4:
      value = value and p > pos
    for p in pos5:
      value = value and p > pos
  # Check if the pattern appears in a "public" or "private" declaration.                                         
  if value:
    if re.match ("^[ ]*(public|private)[\s\S]+(function|subroutine)", line, re.IGNORECASE):                                                  
      value = False 
  
  # "function" can be part of a subroutine name which is called, e.g. "call foo_function".
  if value:
    if re.match ("^[ ]*call[\s\S]*function", line, re.IGNORECASE):
      value = False
    
  return value

def split_alloc_argument (arg, ignore_percent=False):
  n_open_brackets = 0
  # If there are % in the string, Ignore any bracket information until each has been encountered.
  all_args = []
  tmp = ""
  for char in arg:
    if not ignore_percent and char == "%":
      tmp += char
    elif n_open_brackets == 0 and char == ",":
      all_args.append(tmp)
      tmp = ""
    else: 
      if char == "(":
        n_open_brackets += 1
      elif char == ")":
        n_open_brackets -= 1 
      tmp += char
  all_args.append(tmp)
  return all_args

def split_line (tot_string, is_alloc, is_dealloc):
   # Remove white spaces and ampersands (&)
   tot_string = re.sub(r"\s+", "", tot_string) # This also removes intermediate newlines, for some reason
   tot_string = re.sub(r"&+", "", tot_string)
   # Get everything in between the outer brackets (...)
   tot_string = tot_string[tot_string.find("(")+1:-1]
   # Get all the fields
   fields = split_alloc_argument (tot_string)
   return fields

def extract_name (field):
  n_open_brackets = 0
  i_found = -1
  for i, c in enumerate(reversed(field)):
    if c == ")":
      n_open_brackets += 1
    elif c == "(":
      n_open_brackets -= 1
    if n_open_brackets == 0:
      i_found = i
      break
  i_found = len(field) - i_found
  return i_found

def construct_vftrace_allocate_call (field):
  # Input: name(n1,n2,... 
  # We obtain the field name by taking everything before the first opening bracket.
  # The array dimensions can be given by functions, e.g. ALLOCATE (arr(f(x))). 
  # Therefore, it is important to only match the first bracket, not proceeding ones, which would be
  # done with a call to "split". 
  first_significant_bracket = extract_name (field)
  name = field[0:first_significant_bracket-1]
  rest = field[first_significant_bracket:-1]
  # Split at the commas. The first element is "<name>(<first_dim>", so we split again at the bracket.
  dims = split_alloc_argument (rest, ignore_percent=True)
  # Create the string which computes the total number of elements in the field by multiplying the individual dimensions.
  dim_string = ""
  for i, dim in enumerate(dims):
    # If there is a colon ("x1:x2") in the string, the dimension size is x2 - x1 + 1.
    # Otherwise, the dimension bounds start at 1, and the size is dim.
    pos1 = dim.find(":")
    if pos1 >= 0:
      # Check if the colon is enclosed by brackets
      pos2 = -1
      pos3 = -1
      for c in range(pos1, 0, -1):
        if dim[c] == ")":
          break
        elif dim[c] == "(":
          pos2 = c
          break
      for c in range(pos1, len(dim)):
        if dim[c] == "(":
          break
        elif dim[c] == ")":
          pos3 = c
          break
      enclosed = pos2 >= 0 and pos3 >= 0 and pos2 < pos1 and pos1 < pos3
      if not enclosed:
        tmp = dim.split(":")
        # Add extra brackets around the second summand to keep correct sign of the expression in there.
        dim_string += "(" + tmp[1] + "-(" + tmp[0] + ")+1)"
      else:
        dim_string += dim
    elif "+" in dim or "-" in dim:
      dim_string += "(" + dim + ")"
    else:
      dim_string += dim
    if i + 1 < len(dims):
      dim_string += "*"
#  return "call vftrace_allocate(\"" + name + "\", " + dim_string + ", storage_size(" + name + ")/8)\n"
  return "call vftrace_allocate(\"" + name + "\", int(" + dim_string + ",int64), storage_size(" + name + ")/8)\n"

def construct_vftrace_deallocate_call (field):
  # The input is simply the field name.
  return "call vftrace_deallocate(\"" + field + "\")\n"

def remove_trailing_comment(line):
  first_1 = line.find("!")
  p = re.search(r"\S",line)
  if p is not None:
    first_2 = p.start()
  else:
    first_2 = 0
  if first_1 > first_2:
    return line[0:first_1]
  else:
    return line

def remove_trailing_semicolon(line):
  i = line.find(";")
  if i >= 0:
    return line[0:i]
  else:
    return line

def line_to_be_continued(line):
  if line.isspace():
    return True
  else:
    line_wo_spaces = re.sub(r"\s+", "", line)  
    if line_wo_spaces != "":
      tmp = remove_trailing_comment(line_wo_spaces)
      last_ampersand = tmp[-1] == "&"
    else:
      last_ampersand = False
    has_leading_comment = re.match("^!", line_wo_spaces)
    has_leading_preprocessor = re.match("^#", line_wo_spaces)
    return (last_ampersand or (has_leading_comment != None) or (has_leading_preprocessor != None))

with open(filename_in, "r") as f_in, open(filename_out, "w") as f_out:
  all_lines = f_in.readlines()
  n_lines = len(all_lines)
  subroutine_start = False
  subroutine_end = False
  skip_subroutine = False
  already_wrapped = all_lines[0] == vftrace_wrapper_marker
  for i_line, line in enumerate(all_lines):
    if i_line == 0 and not already_wrapped:
      f_out.write (vftrace_wrapper_marker)

    has_leading_comment = re.match("^[ ]*!", line)
    if has_leading_comment or already_wrapped:
       f_out.write (line)
       continue
    is_alloc = allocate_pattern.match(line)
    is_dealloc = deallocate_pattern.match(line)
    # Put "use vftrace" after every subroutine or function definition, regardless if it is actually used.
    is_function_or_subroutine = check_if_function_or_subroutine (line)

    # We need to find out when the function definition has been written completely in the previous iteration.
    # On flag indicates that the subroutine definition has been started. If it is set, we check if it is finished.
    # If the latter one is set before all the other ones, it's time to insert the use statement.
    if subroutine_end and not skip_subroutine:
       f_out.write ("use vftrace\n")
       f_out.write ("use iso_fortran_env, only: int64\n")
       subroutine_end = False
    if is_function_or_subroutine:
       if re.search("pure ", line, re.IGNORECASE) or re.search("elemental ", line, re.IGNORECASE):
         skip_subroutine = True
       else:
         skip_subroutine = False
         subroutine_start = True
    if subroutine_start:
       if not line_to_be_continued(line):
         subroutine_end = True
         subroutine_start = False

    # Register allocate and deallocate calls.
    if not skip_subroutine and (is_alloc or is_dealloc):
      # Concatenate line breaks indicated by ampersands "&"
      line_tmp = line
      tot_string = remove_trailing_comment(line_tmp)
      tot_string = remove_trailing_semicolon(tot_string)
      i = i_line
      while line_to_be_continued(line_tmp):
        i = i + 1

        if i < n_lines:
          line_tmp = all_lines[i]
        else:
          break
        if not re.match("^[ ]*!", line_tmp): # Not an entire comment line
          tot_string += remove_trailing_comment(line_tmp)
      fields = split_line(tot_string, is_alloc, is_dealloc)
      fields_clear = []
      for f in fields:
        if f[-1] == ")" and not "stat=" in f:
          fields_clear.append(f)
      for field in fields_clear:
        if is_alloc:
          f_out.write (construct_vftrace_allocate_call(field))
        elif is_dealloc:
          f_out.write (construct_vftrace_deallocate_call(field))

    f_out.write (line)
