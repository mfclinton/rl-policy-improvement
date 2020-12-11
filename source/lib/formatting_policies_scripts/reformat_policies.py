#Used to reformat policies to corecting numbering schema
import os
directory = r"text_policies/pdis_2/"
id = 81
for filename in os.listdir(directory):
    shared_name = "policy"
    idx = filename.find(shared_name)
    if(idx != -1):
        start_index = filename[idx + len(shared_name)]
        end_index = filename.index(".")
        old_dir = directory + filename
        new_dir = directory + shared_name + str(id) + ".txt"
        os.rename(old_dir, new_dir)
        id += 1