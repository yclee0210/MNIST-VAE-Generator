import os
import numpy as np
import itertools

import config

def calculate_l2_distance(reference, sample):
    norm_val = 0.0
    ref_flat = reference.flatten()
    sample_flat = sample.flatten()

    norm_val = np.linalg.norm(reference - sample, ord=2)
    #norm_val = np.linalg.norm(ref_flat - sample_flat, ord=2)

    return (norm_val)

arr = range(0,1000)
combs = itertools.combinations(arr,2)
combs_list = list(combs)
num_combs = len(list(combs))

sampling = config.SAMPLE_COUNT
working_dir = config.WORKING_DIR + sampling + "/"

print ("Sampling: ", sampling)
stdev_folder_arr = os.listdir(working_dir)

for stdev_folder in stdev_folder_arr:
    std_dev = stdev_folder.split("_")[1]
    print ("Std Dev: ", std_dev)
    print ("Label, L2 Distance")

    label_folder_arr = os.listdir(working_dir + stdev_folder)
    for label_folder in label_folder_arr:
        if (label_folder == 'samples.png'):
            continue
        label = label_folder.split("_")[1]
        contents = os.listdir(working_dir + stdev_folder + "/" + label_folder)
        for file in contents:
            filepath = stdev_folder + "/" + label_folder + "/" + file
            X = np.load(working_dir + filepath)
            X_flat = X.reshape(1000,784)
            l2_dist_arr = []
            for combination in combs_list:
                index1 = combination[0]
                index2 = combination[1]
                X1 = X_flat[index1]
                X2 = X_flat[index2]
                cur_dist = calculate_l2_distance(X1,X2)
                l2_dist_arr.append(cur_dist)
            mean_dist = np.mean(l2_dist_arr)
            print(label,",",mean_dist)