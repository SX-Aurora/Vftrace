#!/usr/bin/env python

import sys
import re

filename_in = sys.argv[1]
filename_out = filename_in + ".vftr"
allocate_pattern = re.compile("^[ ]*allocate[\ ,\(]", re.IGNORECASE)
deallocate_pattern = re.compile("^[ ]*deallocate[\ ,\(]", re.IGNORECASE)
subroutine_pattern = re.compile(r"^[ ]*(pure)?(elemental)?[ ]*subroutine[\s]+[\S]", re.IGNORECASE)
function_pattern = re.compile(r"^[ ]*(pure)?(elemental)?[ ]*function[\s]+[\S]", re.IGNORECASE)

#def split_alloc_argument (arg, check_any_bracket=False):
def split_alloc_argument (arg, ignore_percent=False):
  #open_bracket = False
  n_open_brackets = 0
  # If there are % in the string, Ignore any bracket information until each has been encountered.
  if not ignore_percent:
    n_percent = arg.count("%")
  else:
    n_percent = 0
  #any_full_bracket = not check_any_bracket
  all_args = []
  tmp = ""
  for char in arg:
    if not ignore_percent and char == "%":
      n_percent -= 1
      tmp += char
    #elif not open_bracket and any_full_bracket and char == ",":
    elif n_percent == 0 and n_open_brackets == 0 and char == ",":
      all_args.append(tmp)
      tmp = ""
      #any_full_bracket = n_percent == 0 and not check_any_bracket
    else: 
      if char == "(":
        n_open_brackets += 1
      elif char == ")":
        #if open_bracket: any_full_bracket = n_percent == 0
        n_open_brackets -= 1 
        #open_bracket = False
      tmp += char
  #if any_full_bracket: all_args.append (tmp)
  all_args.append(tmp)
  return all_args

def split_line (tot_string, is_alloc, is_dealloc):
   # Remove white spaces and ampersands (&)
   tot_string = re.sub(r"\s+", "", tot_string) # This also removes intermediate newlines, for some reason
   tot_string = re.sub(r"&+", "", tot_string)
   # Get everything in between the outer brackets (...)
   tot_string = tot_string[tot_string.find("(")+1:-1]
   #print ("Original: ", tot_string)
   # Get all the fields
   #print ("Before splitting: ", tot_string)
   #print ("tot_string: ", tot_string)
   fields = split_alloc_argument (tot_string)
   print ("Fields: ", fields)

   #if is_alloc:
   #  fields = tot_string.split(",")
   #elif is_dealloc:
   #  fields = tot_string.split(",")
   #print ("After splitting: ", fields)
   # Check if there is a "STAT" argument in the last element of the list. If so, we return the list reduced by one element.
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
  print ("FIELD: ", field)
  first_significant_bracket = extract_name (field)
  name = field[0:first_significant_bracket-1]
  #percent_indices = [i for i, this_char in enumerate(field) if this_char == "%"]
  #n_percent = len(percent_indices)
  #bracket_indices = [i for i, this_char in enumerate(field) if this_char == "("]
  #print ("percent_indices: ", percent_indices)
  #print ("bracket_indices: ", bracket_indices)
  #first_significant_bracket = bracket_indices[0]
  #if n_percent > 0:
  #  for br_index in bracket_indices:
  #    if br_index > percent_indices[0]:  
  #      first_significant_bracket = br_index
  #      break
  #name = field[0:first_significant_bracket]
  rest = field[first_significant_bracket:-1]
  print ("NAME: ", name, "REST: ", rest)
  # Split at the commas. The first element is "<name>(<first_dim>", so we split again at the bracket.
  dims = split_alloc_argument (rest, ignore_percent=True)
  print ("dims: ", dims)
  # Create the string which computes the total number of elements in the field by multiplying the individual dimensions.
  dim_string = ""
  for i, dim in enumerate(dims):
    # If there is a colon ("x1:x2") in the string, the dimension size is x2 - x1 + 1.
    # Otherwise, the dimension bounds start at 1, and the size is dim.
    if ":" in dim:
      tmp = dim.split(":")
      #print ("tmp: ", tmp)
      dim_string += "(" + tmp[1] + "-" + tmp[0] + "+1)"
    elif "+" in dim or "-" in dim:
      dim_string += "(" + dim + ")"
    else:
      dim_string += dim
    if i + 1 < len(dims):
      dim_string += "*"
  return "call vftrace_allocate(\"" + name + "\", " + dim_string + ", storage_size(" + name + ")/8)\n"

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
  #first_2 = re.search(r"\S",line).start()
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
    return (last_ampersand or has_leading_comment or has_leading_preprocessor)

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
      #while "&" in line_tmp: 
      while line_to_be_continued(line_tmp):
        i = i + 1

        if i < n_lines:
          line_tmp = all_lines[i]
        else:
          break
        tot_string += remove_trailing_comment(line_tmp)
      fields = split_line(tot_string, is_alloc, is_dealloc)
      fields_clear = []
      for f in fields:
        if f[-1] == ")" and not "stat=" in f:
          fields_clear.append(f)
      print ("After dropping: ", fields_clear)
      #print ("FIELDS: ", fields)
      for field in fields_clear:
        if is_alloc:
          f_out.write (construct_vftrace_allocate_call(field))
        elif is_dealloc:
          f_out.write (construct_vftrace_deallocate_call(field))

    f_out.write (line)
