# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 22:58:45 2022

@author: vipro
"""

import os.path
import numpy as np
import json
import pandas as pd



def ratio_viability(live_value, dead_value, live_range = [0.015, 0.09], dead_range = [0.08, 0.02], resolution = 0.01): #### ranges [c_0, c_100]

    value = live_value/dead_value
    
    step = int(100/resolution)
    live = np.linspace(live_range[0], live_range[1], num=step)
    dead = np.linspace(dead_range[0], dead_range[1], num=step)
    ratio = np.divide(live,dead)
    
    if (live_value < live_range[0]) and (dead_value < dead_range[0]):
        viability = -1
    else:
        index = -1
        #### search through the table data
        for i in range(step):
            if (ratio[i]-value) >= 0:
                index = i
                break

        viability = index/step*100

        if (index == -1) and (value < ratio[0]):
            viability = 0

        if (index == -1) and (value > ratio[step-1]):
            viability = 101 
    
    return viability



#Defining objective function (Hill equation)
def hill(c, e1, h, ec):
    return 1+np.divide(e1-1,np.power(np.divide(ec,c),h)+1) 

def norm_conc(x0, norm_bounds):
    try:
        norm_bounds_log = np.log10(norm_bounds)
        x0_log = np.log10(x0)
        x_scaled = (x0_log - min(norm_bounds_log))/(max(norm_bounds_log) - min(norm_bounds_log))
    except:
        x_scalled = []
    return x_scaled
    
def norm_conc_inverse(x, norm_bounds):
    norm_bounds_log = np.log10(norm_bounds)
    x_inverse = (x*(max(norm_bounds_log) - min(norm_bounds_log)))+min(norm_bounds_log)
    x_scaled = np.power(10,x_inverse)
    return x_scaled

def cell_viability(x, y, z, fit_x, fit_y, fit_z):
        
    cv_x = hill(x, fit_x[0], fit_x[1], fit_x[2])
    cv_y = hill(y, fit_y[0], fit_y[1], fit_y[2])
    cv_z = hill(z, fit_z[0], fit_z[1], fit_z[2])
   
    cv = cv_x*cv_y*cv_z
    return cv 

def norm_data_inverse(data_raw,fit_dict):
    data = data_raw.copy()
    data['conc0_inv'] = norm_conc_inverse(data['conc0'],fit_dict['bounds'][0])
    data['conc1_inv'] = norm_conc_inverse(data['conc1'],fit_dict['bounds'][1])
    data['conc2_inv'] = norm_conc_inverse(data['conc2'],fit_dict['bounds'][2])
    return data

def dilution(conc,stock, vol):
    v = round(vol*conc/stock,1)
    return v

def dil_table(data, stock, vol = 1000):
    data_dil = data.copy()
    data_dil['vol0'] = dilution(data_dil['conc0_inv'],stock[0],vol)
    data_dil['vol1'] = dilution(data_dil['conc1_inv'],stock[1],vol)
    data_dil['vol2'] = dilution(data_dil['conc2_inv'],stock[2],vol)
    data_dil['vol_media'] = vol - data_dil['vol0'] - data_dil['vol1']- data_dil['vol2']

    return data_dil

def conc_total(x,y,z):
    d = np.sqrt(np.square(x) + np.square(y) + np.square(z))
    return d



def filelist(folder, ext):
    try:
        # Get list of files in folder
        file_list = os.listdir(folder)
    except:
        file_list = []
    fnames = [
        f
        for f in file_list
        if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith((ext))
    ]
    return fnames

def paste_to_clipboard(values = []):
    pd.DataFrame(columns = values).to_clipboard(index= False) 


            
def load_settings(name):
    with open(name) as f:
        data = json.load(f)
    return data

def save_settings(name, fit_data):
    with open(name, 'w') as json_file:
        json.dump(fit_data, json_file)


def main():
    pass


if __name__ == "__main__":
    main()