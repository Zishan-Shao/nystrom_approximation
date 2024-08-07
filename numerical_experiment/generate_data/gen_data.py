import os
import numpy as np
import random

save_path = '../data/synthetic/' # this is the relative path to the datafile
                      # NOTE: "data" folder should be in same folder of ./generate_data 
filename = "syn_data_large_20000_20.txt" # You define it here
#file_fn = input("Enter the file name: ")
#filename = file_fn + ".txt" # you can also input your filename if you want

#print(filename)

## Hyperparameters
m = 20000; # number of observations (rows)
n = 20; # number of features (cols)
nnz =m*n; # total number of non-zero values to be considered
          # NOTE: this number should never exceed m*n
num_cls = 10; 


row_dense = nnz // m; # average density of the row


## Error Checking
if (m*n < nnz):
    print(f"Errno: nnz should never exceed m*n: {m*n}, current: {nnz}")
    exit(0)

## Main loop
try: 
    with open(os.path.join(save_path, filename), "w") as file:
        # file.write(str(contents))        # writes a string to a file
        for i in range(m):
            # create the label, maximum num_cls
            lb = random.randint(0,num_cls-1)
            file.write(str(lb))
            
            # Generate non-zero indices
            row_entry = sorted(random.sample(range(n), row_dense))
            
            # inner loop: create the observation
            #             The observation was in form: feature:value
            for idx in row_entry:
                value = random.uniform(-1.0, 1.0)
                obs = f" {idx}:{value}"
                file.write(obs)
                #print("haha")
            file.write("\n")
except:
    print("Errno: Data Generation Error, Pls contact Zishan or ChatGPT (More prestigious than Zishan)")
    
file.close()

