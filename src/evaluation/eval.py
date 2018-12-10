import sys
import os

import numpy as np

import config

def main():
    if (len(sys.argv) != 3):
        print ("Enter the directory for your samples and\
            specified label as 1st and second arguments.")
    
    sample_dir = sys.argv[1]
    label = sys.argv[2]
    

    reference_dir = os.path.dirname(__file__) + config.REFERENCE_DIR
    reference_arr = np.genfromtxt(reference_dir + '/' + str(label) + '.csv', delimiter=',')
    
    norm_arr = []
    cosine_arr = []

    sample_folder = os.listdir(sample_dir)
    for file in sample_folder:
        cur_filename = sample_dir + "/" + file
        cur_arr = np.genfromtxt(cur_filename, delimiter=',')
        
        cur_norm = calculate_l2_norm(reference_arr, cur_arr)
        cur_cos = calculate_cosine_similarity(reference_arr, cur_arr)

        print ("Sample file: ", file)
        print ("   L2 Norm Distance: ", cur_norm)
        print ("   Cosine Similarity: ", cur_cos)
        norm_arr.append(cur_norm)
        cosine_arr.append(cur_cos)

    norm_mean = np.mean(norm_arr)
    norm_stdev = np.std(norm_arr)

    cos_sim_mean = np.mean(cosine_arr)
    cos_sim_stdev = np.std(cosine_arr)

    print ("L2 Norm Distance")
    print ("   Mean: ", norm_mean)
    print ("   Std Dev: ", norm_stdev)

    print ("Cosine Similarity")
    print ("   Mean: ", cos_sim_mean)
    print ("   Std Dev: ", cos_sim_stdev)
        

def calculate_l2_norm(reference, sample):
    norm_val = 0.0
    ref_flat = reference.flatten()
    sample_flat = sample.flatten()

    norm_val = np.linalg.norm(reference - sample, ord=2)
    return (norm_val)

def calculate_cosine_similarity(reference, sample):
    ref_flat = reference.flatten()
    sample_flat = sample.flatten()

    similarity = np.dot(ref_flat, sample_flat) / (np.linalg.norm(ref_flat, ord=2) * np.linalg.norm(sample_flat, ord=2))
    return (similarity)

main()
    