#!/usr/bin/env python

"""
@author: pritesh-mehta
"""
import multiprocessing

import numpy as np
from scipy.optimize import curve_fit
from pathlib import Path
from argparse import ArgumentParser
import itertools as it
from tqdm.auto import tqdm

from dwi_utilities.monoexponential_decay import log_func, func
import dwi_utilities.nifti_utilities as nutil

def comp_high_b_case(case_dir, target_bval, save_case=False, output_dir=None, extension='.nii.gz'):
    """Generate high b-value DWI using low b-value DWI (case)
    """
    eps = 1e-8 
    
    data_stack = []
    bval_list = []
    filepaths = nutil.path_generator(case_dir)
    for path in filepaths:
        name, nii, data = nutil.load(path)
        data_stack.append(data)
        bval_list.append(name.replace('.nii.gz','').replace('b',''))
        
    # order data stack in order of ascending b-value
    # bval_list, data_stack = \
    #     zip(*sorted(zip(bval_list, data_stack))) # not working as list values are string and ascending order is not correct as they need to be in integer
        
    # # generate high b-value
    # bval_list = np.array(bval_list)
    # data = np.array(data_stack)
        
    bval_list = np.array(bval_list).astype(int)
    bval_list, data_stack = \
        zip(*sorted(zip(bval_list, data_stack))) # not working as list values are string and ascending order is not correct as they need to be in integer
    # print(bval_list)
    # generate high b-value
    data = np.array(data_stack) # data is in shape (num, W, H, D)
    shape = np.shape(data[0])
    # print(shape)
    highb_data = np.zeros(shape)

    pixels = it.product(range(shape[0]), range(shape[1]), range(shape[2]))
            
    for i in range(shape[0]): # get ith pixel
        for j in range(shape[1]): # get jth pixel
            for k in range(shape[2]): # get each slice
                y = []
                for array in data:  # loop through image in data array
                    y.append(array[i][j][k]) # get pixel for position in [i][j] of slice k for each image in data array
                x = bval_list
                y = np.array(y) + eps
                z = np.log(y)
                popt, pcov = curve_fit(log_func, x, z)
                if popt[1] < 0:
                    highb_data[i][j][k] = 0
                else:
                    highb_data[i][j][k] = func(target_bval, np.exp(popt[0]), popt[1])    


    # with multiprocessing.Pool(processes=1) as pool:
    #     pool.starmap(fit_values, (pixels, data, it.repeat(bval_list), log_func, func, it.repeat(target_bval))) 
    # for pixel in tqdm(list(pixels), total=len(list(pixels))):
    #     highb_data = fit_values(pixel, data, bval_list, log_func, func, target_bval, highb_data)

    if save_case:
        case_name = Path(case_dir).parts[-1]
        save_path = Path(output_dir) / (case_name + extension)
        nutil.save(save_path, nii, highb_data)
    
    return highb_data

def fit_values(pixels, data, x, log_func, func, target_bval, highb_data):
    i, j, k = pixels
    eps = 1e-8
    y = data[:, i, j, k]
    y = np.array(y) + eps
    y = np.log(y)
    popt, _ = curve_fit(log_func, x, y)

    if popt[1] < 0:
        highb_data[i][j][k] = 0
    else:
        print("true")
        highb_data[i][j][k] = func(target_bval, np.exp(popt[0]), popt[1])   
    return highb_data

def comp_high_b_dir(cases_dir, target_bval, output_dir, extension='.nii.gz'):
    """Generate high b-value DWI using low b-value DWI (directory)
    """
    for case_dir in Path(cases_dir).iterdir():
        print("Processing:", case_dir)
        comp_high_b_case(case_dir, target_bval, save_case=True, output_dir=output_dir, extension=extension)
    return None

def process():
    parser = ArgumentParser()
    parser.add_argument('--input_dir', required=True, type=str)
    parser.add_argument('--target_bval', required=True, type=int)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--case', required=False, action="store_true")
    parser.add_argument('--extension', required=False, type=str, default='.nii.gz')
      
    args = parser.parse_args()
    
    if args.case:
        comp_high_b_case(args.input_dir, args.target_bval, save_case=True, output_dir=args.output_dir, 
                extension=args.extension)
    else:
        comp_high_b_dir(args.input_dir, args.target_bval, args.output_dir,
                 extension=args.extension)
    
if __name__ == "__main__":
    process()
