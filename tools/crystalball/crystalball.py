#!/usr/bin/env python
import sys

import read_functions
import progression

if len(sys.argv) > 1:
  all_dicts = []
  for filename in sys.argv[1:]:
    top_5_stack_ids, func_dict = read_functions.create_dictionary(filename)
    all_dicts.append(func_dict) 
    
    print ("top_5 in " + filename + ": ")
    for i in top_5_stack_ids:
      print (func_dict[i])
    print ("**************************************************")

  global_dict = read_functions.synchronize_dictionaries(all_dicts) 
  for this_hash, prog_entry in global_dict.items():
    progression.determine_progression_type(prog_entry)


# Do the prediction
  x_predict = 2.0
  t_predict = 0.0
  for this_hash, prog_entry in global_dict.items():
    if not isinstance(prog_entry.progression_type, progression.prog_undefined):
      t_predict += prog_entry.progression_type.predict(x_predict)

  print ("Prediction: ", t_predict)

else:
  print ("Require at least one log file as argument!")
