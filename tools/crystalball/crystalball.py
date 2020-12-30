#!/usr/bin/env python
import sys

import tkinter as tk

import read_functions
import progression

def do_prediction():

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

# Do the prediction
    x_predict = float(entry_entry.get())
    print ("Predict for: ", x_predict)
    t_predict = 0
    for this_hash, prog_entry in global_dict.items():
      progression.determine_progression_type(prog_entry, len(sys.argv) - 1)
      if not isinstance(prog_entry.progression_type, progression.prog_undefined):
         t_predict += prog_entry.progression_type.predict(x_predict)

    print ("Prediction: ", t_predict)
  else:
    print ("Require at least one log file as argument!")



window = tk.Tk()
window.title ("The crystal ball performance predictor")

frame_entry = tk.Frame (master = window)
entry_label = tk.Label (master = frame_entry, text = "Runtime:")
entry_entry = tk.Entry (master = frame_entry, width=5)
entry_button = tk.Button (master = frame_entry, text = "Predict", command = do_prediction)
entry_label.grid (row=0, column=0)
entry_entry.grid (row=0, column=1)
entry_button.grid (row=0, column=2)
frame_entry.grid (row=0, column=0)

window.mainloop()


