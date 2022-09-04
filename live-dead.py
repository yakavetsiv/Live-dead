# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 21:57:11 2022

@author: vipro
"""

import PySimpleGUI as sg
import os.path
import cv2
import numpy as np
import json
import pandas as pd


from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage import io
from skimage.filters import threshold_triangle

from utils.image_processing import *
from utils.utils import *
from utils.plots import *


settings = {
    'folder' : '/Users/viprorok/Google Диск/Laba/Github/Im_processing/Synergy index/',
    'invert_flag' : True,
    'autocontrast_flag' : True,
    'scaling_factor': {'100 um': 6, '200 um': 12, '300 um': 12},       ### 6 for 100um and 12 for 200 um and 300 um
    'rows' : 4,
    'columns' : 25,
    'radius' : {'100 um': 30, '200 um': 55, '300 um': 85},     ### [30 for 100 um] and [85 for 300um] [55 for 200um]
    'offset' : 10,
    'flag_100' : False,
    'flag_300' : True,
    'flag_200': False,
    'devices' : 1,
    'device_types':['100 um','200 um','300 um'],
    'default_device_type': 2,
    'cell_types': ['MCF-7', 'DCBXTO.58'],
    'default_cell_type': 0,
    'live_range' : {'MCF-7' : [0.015, 0.09], 'DCBXTO.58': [0.012, 0.085]}, ##  [0.015, 0.09] for MCF-7 cells ; [0.012, 0.085] for .58
    'dead_range' : {'MCF-7' : [0.08, 0.02], 'DCBXTO.58': [0.085, 0.02]}, ## [0.08, 0.02] for MCF-7 cells ; [0.085, 0.02] for .58
    'hist_flag' : True,
    'ch_flag' : False,
    'dil_flag': True,
    'cv_threshold': 0.5,
    'synergy_flag': True,
    'cv_flag': False, 
    '3D_text': True,
    '3D_alpha': 0.5,
    'conc_flag': False
    }

boxes = []
cords = []  

image_bf_raw = 0
image_live_raw = 0 
image_dead_raw = 0 
image_bf = 0
image_live = 0
image_dead = 0
num_channels = 0
window =[]
figure_canvas_agg = None

flag_crop = False
img_file_flag = False
flag_grid = False
flag_seg = False
calc_flag = False


device_type = settings['device_types'][settings['default_device_type']]
cell_type = settings['cell_types'][settings['default_cell_type']]

#### merging of the brightfield image and the defined regions

def draw_regions(cords, image, radius, window):
        
    
    window['-SEG-'].update(len(cords))
    image_bf_s = image.copy()
    
    for cord in cords:
        cent_x, cent_y = cord
        image_bf_s = cv2.circle(image_bf_s, (int(cent_x), int(cent_y)), radius, [0], 5)
    return image_bf_s 


def on_mouse_crop(event, x, y, flags, params):
    global image_bf
    global image_live, image_dead
    global settings
    global flag_crop
    global boxes
    global device_type
    
    scaling_factor = settings['scaling_factor'][device_type]
    
    #### defining the first coordinate of box using left button click of the mouse
    if event == cv2.EVENT_LBUTTONDOWN:
        sbox = [x, y]
        boxes.append(sbox)
    
    #### defining the second coordinate of box using left button  of the mouse and cropping of the image
    elif event == cv2.EVENT_LBUTTONUP:
        ebox = [x, y]
        boxes.append(ebox)
        
        #### cropping the images
        image_bf = crop(image_bf, boxes[0][1]*scaling_factor, boxes[1][1]*scaling_factor,
                        boxes[0][0]*scaling_factor, boxes[1][0]*scaling_factor)
        
        image_live = crop(image_live, boxes[0][1]*scaling_factor, boxes[1][1]*scaling_factor, 
                             boxes[0][0]*scaling_factor, boxes[1][0]*scaling_factor )
        
        image_dead = crop(image_dead, boxes[0][1]*scaling_factor, boxes[1][1]*scaling_factor, 
                             boxes[0][0]*scaling_factor, boxes[1][0]*scaling_factor )
        
        #### turning off the cropping loop function
        flag_crop = 0
        cv2.setMouseCallback('BF', nothing)
        
        #### show updated brightfield image
        show_img(image_bf, 'BF', scaling_factor)



#### adding/removal of the region using the mouse left button click
def on_mouse_rem(event, x, y, flags, params):
    global cords
    global settings
    global image_bf
    global device_type

    radius = settings['radius'][device_type]
    scaling_factor = settings['scaling_factor'][device_type]
    #### flag for the addition(FALSE)/removal(TRUE) of the region
    flag = 0
    
    if event == cv2.EVENT_LBUTTONDOWN:
        #### search for the existing region containing the chosen pixel
        for i, cord in enumerate(cords):
            cent_x, cent_y = cord
            if (x*scaling_factor > cent_x - radius) and (x*scaling_factor < cent_x + radius):
                if (y*scaling_factor > cent_y - radius) and (y*scaling_factor < cent_y + radius):
                    #### region was found: delete the region from the list and skip the addition of the region
                    flag = 1
                    cords = np.delete(cords, i, axis=0)
                    #### show updated regions
                    
                    image_bf_s = draw_regions(cords, image_bf, radius, window)
                    show_img(image_bf_s, 'Segment', scaling_factor)
                    
        #### region was NOT found: add a new region with the coordinates of the chosen pixel
        if flag == 0:
            l = len(cords)
            cord = np.array([(x*scaling_factor, y*scaling_factor)])
            cords = np.append(cords, cord).reshape(l+1, 2)
            #### show updated regions

            image_bf_s = draw_regions(cords, image_bf, radius, window)
            show_img(image_bf_s, 'Segment', scaling_factor)
            


#### defining the grid using 3 points (1-top left, 2-bottom left, 3-bottom right) of the grid corners
def on_mouse_grid(event, x, y, flags, params):
    global cords 
    global image_bf 
    global settings
    global flag_grid, flag_seg
    global device_type
    
    scaling_factor = settings['scaling_factor'][device_type]
    radius = settings['radius'][device_type]
    devices = settings['devices']
    columns = settings['columns']
    rows = settings['rows']
    
    #### add(FALSE)/remove(TRUE) point from the list
    #### to remove the point we need to left click on it once more time
    flag = 0
    
    if flag_grid == 1:
        if event == cv2.EVENT_LBUTTONDOWN:
            #### if the chosen pixel belongs to th already existing point - detete point
            for i, cord in enumerate(cords):
                cent_x, cent_y = cord
                if (x*scaling_factor > cent_x - radius) and (x*scaling_factor < cent_x + radius):
                    if (y*scaling_factor > cent_y - radius) and (y*scaling_factor < cent_y + radius):
                        flag = 1
                        cords = np.delete(cords, i, axis=0)
                        #### show updated regions
                        image_bf_s = draw_regions(cords, image_bf, radius, window)
                        show_img(image_bf_s, 'Segment', scaling_factor)
                        
            
            #### add the point with the coordinates of the chosen pixel
            if flag == 0:
                l = len(cords)
                cord = np.array([(x*scaling_factor, y*scaling_factor)])
                cords = np.append(cords, cord).reshape(l+1, 2)
                #### if it was the THIRD point (all required coordinates have been chosen) - pass the function
                if ((len(cords) == 3) & (rows > 1)) & (devices == 1) :
                    
                    #### calculation of the coordinates of all the wells in the grid
                    cords = grid(cords[0], cords[1], cords[2], (0,0), rows, columns, devices)
                    flag_grid = 0
                    flag_seg = 1
                
                if ((len(cords) == 4) & (rows > 1)) & (devices > 1) :
                    
                    #### calculation of the coordinates of all the wells in the grid
                    cords = grid(cords[0], cords[1], cords[2], cords[3], rows, columns, devices)
                    flag_grid = 0
                    flag_seg = 1
                    
                if ((len(cords) == 3) & (rows == 1)) & (devices > 1) :
                    
                    #### calculation of the coordinates of all the wells in the grid
                    cords = grid_1ch(cords[0], cords[1], cords[2], columns, devices)
                    flag_grid = 0
                    flag_seg = 1
                    
                if ((len(cords)==2) & (rows == 1)) & (devices == 1):
                    
                    #### calculation of the coordinates of all the wells in the grid
                    cords = grid_1ch(cords[0], cords[1], (0,0), columns, devices)
                    flag_grid = 0
                    flag_seg = 1
                
                
                #### show updated regions

                image_bf_s = draw_regions(cords, image_bf, radius, window)
                show_img(image_bf_s, 'Segment', scaling_factor)
                


def main():
    global settings
    global img_file_flag, flag_crop
    global image_bf_raw, image_live_raw, image_dead_raw, image_bf, image_live, image_dead, num_channels
    global boxes, cords
    global flag_grid, flag_seg
    global window
    global calc_flag
    global device_type 
    global cell_type   
    global figure_canvas_agg
    
    img_filename = ''
    csv_filename = ''
    d1_name = ''
    d3_name = ''
    data_name = ''
    fit_name = ''
    img_flag = 0
    text_channels = ''
    rotation = 0
    report_flag = False
    f_flag = False
    

    
    
    
    layout = [[sg.Button('Analysis of images')],     
                      [sg.Button('Relative cell viability'), sg.Button('Correct CV')],
                      [sg.Button('Synergy')],
                      [sg.Text()],
                      [sg.Button('Settings'),sg.Button('Quit')]]
    
    window = sg.Window('Main menu', layout)

    while True:
        event, values = window.read()

        if event == 'Quit' or event == sg.WIN_CLOSED :
            window.close()
            cv2.destroyAllWindows()
            break
        
        elif event == 'Main menu':
            window.close()
            cv2.destroyAllWindows()
            layout = [[sg.Button('Analysis of images')],     
                      [sg.Button('Relative cell viability'), sg.Button('Correct CV')],
                      [sg.Button('Synergy')],
                      [sg.Text()],
                      [sg.Button('Settings'),sg.Button('Quit')]]
            window = sg.Window('Main menu', layout)
        
        elif event == 'Analysis of images':
            window.close()
            cv2.destroyAllWindows()
            
            fnames = filelist(settings['folder'], ".tif")

            layout_files = [
                [
                    sg.Text("Image Folder"),
                    sg.In(settings['folder'],size=(35, 1), enable_events=True, key="-FOLDER-"),
                    sg.FolderBrowse(initial_folder=settings['folder']),
                ],
                [
                    sg.Listbox(values=fnames, enable_events=True, size=(50, 10), key="-FILE LIST-")
                ],
                [
                    sg.Text(size=(40, 1), key="-TOUT-")
                ],
                [
                    sg.Checkbox('BF = invert fluo', default=settings['invert_flag'], key="-INVERT-"),
                    sg.Checkbox('Autocontrast', default=settings['autocontrast_flag'], key="-AUTOCONTRAST-"),
                    sg.Button('Load image'),
                ]]
            
            layout_analysis = [
                [
                    sg.Text(size=(40, 1), key="-CHANNELS-")
                ],
                [
                    sg.pin(sg.Button('Edit images', visible = True)), 
                    sg.Button('Grid'),
                    sg.Button('Add/Remove segments'),
                    sg.Txt(size=(8,1), key='-SEG-'),
                    sg.Button('Calculate')
                ]]
            layout_buttons = [[
                    sg.Button('Main menu'),
                    sg.Button('Settings'),
                    sg.Button('Quit')
                ]]
            
            layout_left = [
                    [sg.Canvas(key='-CANVAS-',size=(50, 50))], 
                    [sg.Radio('Histogram', 1,key = "-HIST-", default=True),
                     sg.Radio('Ch1 vs Ch2', 1,key = "-CH-"),
                     sg.Button('Save plot'), 
                     sg.Button('Save results'), 
                     sg.Button('Save images')]
                            ]

            layout_right = layout_files + [[sg.pin(sg.Column(layout_analysis, key='-ANALYSIS-', visible=False, pad = (0,0)))]] + layout_buttons
            
            
            layout = [[sg.Column(layout_right), sg.pin(sg.Column(layout_left, key='-PLOT-', visible = False))]]            


            window = sg.Window('Please choice the image file (*.tif)', layout, finalize = True)
            
            if img_file_flag:
                window['-ANALYSIS-'].update(visible = True)
                window['-CHANNELS-'].update(text_channels)
                window.TKroot.title(img_filename)

                if flag_seg:
                    radius = settings['radius'][device_type]
                    image_bf_s = draw_regions(cords, image_bf, radius, window)
                    if calc_flag:              
                        window['-PLOT-'].update(visible = True)
                        if settings['hist_flag']:
                            figure, figure_canvas_agg = plot_hist(df_stats, 30, window['-CANVAS-'].TKCanvas, figure_canvas_agg)
            
              
        elif event == "-FOLDER-":   # New folder has been chosen
            settings['folder'] = values["-FOLDER-"]
            fnames = filelist(settings['folder'], ".tif")
            window["-FILE LIST-"].update(fnames)
            
        elif event == "-FILE LIST-":  # A file was chosen from the listbox
            try:
                img_filename = os.path.join(values["-FOLDER-"], values["-FILE LIST-"][0])
                window["-TOUT-"].update(img_filename)
                img_flag = 1
            except:
                pass
         
        
        elif event == "Correct CV":
            window.close()
            fnames = filelist(settings['folder'], ".csv")

            layout_files = [
                [
                    sg.Text("Image Folder"),
                    sg.In(settings['folder'],size=(35, 1), enable_events=True, key="-FOLDER_CORRECT-"),
                    sg.FolderBrowse(initial_folder=settings['folder']),
                ],
                [
                    sg.Listbox(values=fnames, enable_events=True, size=(50, 10), key="-FILE LIST_CORRECT-")
                ],
                [
                    sg.Text(size=(40, 1), key="-TOUT-")
                ],
                [
                    sg.Button('Correct report'),
                ]]
                
            layout_buttons = [[
                    sg.Button('Main menu'),
                    sg.Button('Settings'),
                    sg.Button('Quit')
                ]]    
            layout = layout_files + layout_buttons
            window = sg.Window('Main menu', layout)
                    
        
        elif event == "Load image":
            
            if (img_filename == '') or (img_flag == False):
                sg.popup('Cancel',"Filename is incorrect")

            else:
                settings['invert_flag'] = values["-INVERT-"]
                settings['autocontrast_flag'] = values["-AUTOCONTRAST-"]
                
                cv2.destroyAllWindows()
                calc_flag = False
                flag_seg = False
                rotation = 0        
                boxes = []    
                flag_seg = 0
                cords = []  
                win_i = 1            
                df_stats = []
                flag_crop = 0
                window['-PLOT-'].update(visible = False)
                window['-ANALYSIS-'].update(visible = False)

                image_raw = io.imread(img_filename)

                if image_raw.ndim == 2:
                    num_channels = 1
                if image_raw.ndim == 3:    
                    num_channels = min(image_raw.shape)
                if settings['invert_flag']:
                    num_channels = num_channels + 1
                    

                if (num_channels == 1):
                    image_bf_raw = image_raw.copy()

                    image_bf_raw = transform.rotate(image_bf_raw, rotation)

                    image_bf = image_bf_raw.copy()

                    
                    image_bf, alpha, beta = automatic_brightness_and_contrast(image_bf)
                    #show_img(image_bf, 'BF', settings['scaling_factor'])
   
                    text_channels = 'Bright'

                if (num_channels == 2):
                    if settings['invert_flag']:
                        image_bf_raw = cv2.bitwise_not(image_raw.copy())
                        thresh = threshold_triangle(image_bf_raw)
                        image_bf_raw = closing(image_bf_raw > thresh, square(1))
                        image_live_raw = image_raw.copy()
                    else: 
                        image_bf_raw = image_raw[0,:,:].copy()
                        image_live_raw = image_raw[1,:,:].copy()

                    image_bf_raw = transform.rotate(image_bf_raw, rotation)
                    image_live_raw = transform.rotate(image_live_raw, rotation)

                    image_bf = image_bf_raw.copy()
                    image_live = image_live_raw.copy()

                    
                    
                    if settings['autocontrast_flag']:
                        image_bf, alpha, beta = automatic_brightness_and_contrast(image_bf)
                        
                    #show_img(image_bf, 'BF', settings['scaling_factor'])

                    text_channels = 'Bright   -   Live'

                if (num_channels == 3):
                    if settings['invert_flag']:
                        image_bf_raw = cv2.bitwise_not(image_raw[0,:,:].copy())
                        thresh = threshold_triangle(image_bf_raw)
                        image_bf_raw = closing(image_bf_raw > thresh, square(1))
                        
                        image_live_raw = image_raw[0,:,:].copy()
                        image_dead_raw = image_raw[1,:,:].copy()
                        
                    else: 
                        image_bf_raw = image_raw[:,:,0].copy()
                        image_live_raw = image_raw[:,:,1].copy()
                        image_dead_raw = image_raw[:,:,2].copy()

                    image_bf_raw = transform.rotate(image_bf_raw, rotation)
                    image_live_raw = transform.rotate(image_live_raw, rotation)
                    image_dead_raw = transform.rotate(image_dead_raw, rotation)

                    image_bf = image_bf_raw.copy()
                    image_live = image_live_raw.copy()
                    image_dead = image_dead_raw.copy()
                    

                    if settings['autocontrast_flag']:
                        image_bf, alpha, beta = automatic_brightness_and_contrast(image_bf)
                        
                    #show_img(image_bf, 'BF', settings['scaling_factor'])
                    
                    text_channels = 'Bright   -   Live   -   Dead'
                    
                if (num_channels == 4):

                    image_bf_raw = cv2.bitwise_not(image_raw[:,:,1].copy())
                    thresh = threshold_triangle(image_bf_raw)
                    image_bf_raw = closing(image_bf_raw > thresh, square(1))

                    image_live_raw = image_raw[:,:,1].copy()
                    image_dead_raw = image_raw[:,:,2].copy()


                    image_bf_raw = transform.rotate(image_bf_raw, rotation)
                    image_live_raw = transform.rotate(image_live_raw, rotation)
                    image_dead_raw = transform.rotate(image_dead_raw, rotation)

                    image_bf = image_bf_raw.copy()
                    image_live = image_live_raw.copy()
                    image_dead = image_dead_raw.copy()
                    
                    cv2.destroyWindow('Segment')

                    if settings['autocontrast_flag']:
                        image_bf, alpha, beta = automatic_brightness_and_contrast(image_bf)
                        
                    #show_img(image_bf, 'BF', settings['scaling_factor'])
                    
                    text_channels = 'Bright   -   Live   -   Dead'
                
                img_file_flag =1
                window['-ANALYSIS-'].update(visible = True)
                window['-CHANNELS-'].update(text_channels)
                window.TKroot.title(img_filename)
                
                if num_channels == 0:
                    sg.popup("Cancel", "Please load the stack with at least 1 channel")
                
        
        
        
        elif event == 'Settings':
            window.close()
            layout = [[sg.Combo(settings['device_types'], default_value = device_type, key='-DEVICE_TYPE-', size=(30, 1))],
                      [sg.Text("Rows"), sg.InputText(settings['rows'], size=(6, 1), key="-Row_input-")],
                      [sg.Text("Columns"), sg.InputText(settings['columns'], size=(6, 1), key="-Columns_input-")],
                      [sg.Text("Devices"), sg.InputText(settings['devices'], size=(6, 1), key="-Devices_input-")],
                      [sg.Combo(settings['cell_types'], default_value = cell_type, key='-CELL_TYPE-', size=(30, 1))],
                      [sg.Button('Update settings'), sg.Button('Load settings'), sg.Button('Save settings')],
                      [sg.Text()],
                      [sg.Button('Main menu'), sg.Button('Analysis of images'), sg.Button('Quit')]]
            window = sg.Window('Main menu', layout)

        elif event == 'Update settings':
            device_type = values['-DEVICE_TYPE-']
            cell_type = values['-CELL_TYPE-']
            settings['columns'] = int(values["-Columns_input-"])
            settings['rows'] = int(values["-Row_input-"])
            settings['devices'] = int(values["-Devices_input-"])
    
        elif event == 'Save settings':            
            save_settings('config.json',settings)
            
        elif event == 'Load settings':
            config = sg.popup_get_file('Enter *.JSON file you wish to process')
            if not config:
                sg.popup("Cancel", "No filename supplied")            
            else:
                path, file = os.path.split(config)
                ext = file.split('.',1)[1]
                if ext == 'json':
                    settings = load_settings(config)
                    window.close()
                    layout = [[sg.Combo(settings['device_types'], default_value = settings['device_types'][settings['default_device_type']], key='-DEVICE_TYPE-', size=(30, 1))],
                              [sg.Text("Rows"), sg.InputText(settings['rows'], size=(6, 1), key="-Row_input-")],
                              [sg.Text("Columns"), sg.InputText(settings['columns'], size=(6, 1), key="-Columns_input-")],
                              [sg.Text("Devices"), sg.InputText(settings['devices'], size=(6, 1), key="-Devices_input-")],
                              [sg.Combo(settings['cell_types'], default_value = settings['cell_types'][settings['default_cell_type']], key='-CELL_TYPE-', size=(30, 1))],
                              [sg.Button('Update settings'), sg.Button('Load settings'), sg.Button('Save settings')],
                              [sg.Text()],
                              [sg.Button('Main menu'), sg.Button('Analysis of images'), sg.Button('Quit')]]
                    window = sg.Window('Main menu', layout)
                    
                    
                else:
                    sg.popup("Cancel", "Incorrect file")

        elif event == 'Edit images':
            window.close()
            layout = [[sg.Text("Rotation"), sg.InputText('0', size=(6, 1), key="Rot_input"), sg.Button('Rotate')],
                      [sg.Button('Crop image'),sg.Button('Autocontrast')],
                      [sg.Text()],
                      [sg.Button('Analysis of images'), sg.Button('Defaults'), sg.Button('Quit')]]
            window = sg.Window('Edit images', layout) 

            show_img(image_bf, 'BF', settings['scaling_factor'][device_type])
            
            
        if event == 'Autocontrast':
            if settings['autocontrast_flag']:
                pass
            else:
                image_bf, alpha, beta = automatic_brightness_and_contrast(image_bf)
                show_img(image_bf, 'BF', settings['scaling_factor'][device_type])


        if event == 'Crop image':
            boxes = []
            flag_crop = 1
            if flag_crop == 1:
                cv2.setMouseCallback('BF', on_mouse_crop)
            else:
                cv2.setMouseCallback('BF', nothing)


        if event == 'Defaults':
            if num_channels == 1:
                image_bf = image_bf_raw
            if num_channels == 2:
                image_bf = image_bf_raw
                image_live = image_live_raw
            if num_channels >= 3:
                image_bf = image_bf_raw
                image_live = image_live_raw
                image_dead = image_dead_raw

            show_img(image_bf, 'BF', settings['scaling_factor'][device_type])

        if event == 'Rotate':
            rotation = float(values["Rot_input"])
            if num_channels == 1:
                image_bf = transform.rotate(image_bf, rotation)

            if num_channels == 2:
                image_bf = transform.rotate(image_bf, rotation)
                image_live = transform.rotate(image_live, rotation)

            if num_channels >= 3:
                image_bf = transform.rotate(image_bf, rotation)
                image_live = transform.rotate(image_live, rotation)
                image_dead = transform.rotate(image_dead, rotation)

            show_img(image_bf, 'BF', settings['scaling_factor'][device_type])
    
    
        if event == 'Grid':
                cv2.destroyAllWindows()  
                image = image_bf.copy()
                cords = []
                flag_grid = True
                radius = settings['radius'][device_type]
                image_bf_s = draw_regions(cords, image_bf, radius, window)
                show_img(image_bf_s, 'Segment', settings['scaling_factor'][device_type])
                #sg.popup("OK", "Please select 3 points on the image using LEFT CLICK\n" +
                #            "1: TOP RIGHT well \n2: BOTTOM RIGHT well \n3: BOTTOM LEFT well \n ")
                cv2.setMouseCallback('Segment', on_mouse_grid, device_type)


        if event == 'Add/Remove segments':
            if flag_seg == 1:
                sg.popup("OK", "Use LEFT CLICK to add/remove the wells")
                cv2.setMouseCallback('Segment', on_mouse_rem)    
        
        if event == 'Save results':
            path, file = os.path.split(img_filename)
            file  = file.split('.',1)[0]
            try:
                if (settings['devices'] == 1):
                    df_stats.to_csv(path  +'/'+ file +'_report.csv',index=True)
                    sg.popup("OK","Saved in" + path)
                if (settings['devices'] > 1):    
                    for i in range(settings['devices']):
                        df = df_stats[df_stats.device == i]
                        df.to_csv(path  +'/'+ file +'_dev'+str(i)+'_report.csv',index=True)
                    sg.popup("OK","Saved " +str(settings['devices']) +" reports in" + path)
            except:
                sg.popup("Cancel", "Error")

        if event == 'Save hist':
                    path, file = os.path.split(img_filename)
                    file  = file.split('.',1)[0]
                    try:
                        figure.savefig(path  +'/'+ file +'_hist.png')
                        sg.popup("Histogram has been saved")
                    except:
                        sg.popup("Cancel", "Error")

        if event == 'Save images':
                    path, file = os.path.split(img_filename)
                    file  = file.split('.',1)[0]
                    try:
                        for i, frame in enumerate(frames):                  
                            io.imsave(path+ '/images/' + str(i) + '.tiff', np.array(frame))                    
                        labels = df_stats.sort_values(by = ["index"], ignore_index = None)['Viability']
                        labels.to_csv(path+ '/images/labels.csv',index=True)

                        sg.popup("Saved " +str(len(frames)) +" images in " + path+ '/images/')

                    except:
                        sg.popup("Cancel", "Error")

        if event == 'Calculate':
            if (img_file_flag == 1) & flag_seg == 1:
                stats, frames = calc_fluo(cords, image_bf, image_live, image_dead, num_channels,settings, device_type, cell_type)
                #### converting the results in the dataframe format

                df_stats = pd.DataFrame(data=stats)

                if num_channels == 1:
                    df_stats.columns = ['x0', 'y0', 'Area','Int_gray']

                if num_channels == 2:
                    df_stats.columns = ['x0', 'y0', 'Area','Int_gray','Int_live']

                if num_channels >= 3:
                    df_stats.columns = ['x0', 'y0', 'Area','Int_gray','Int_live', 'Int_dead', 'Ratio','Viability']


                #### calculation of the position of the well in the grid


                df_stats = df_stats.reset_index()
                num_wells = settings['rows']* settings['columns']
                df_stats['device'] = (df_stats['index'] // num_wells)+1
                df_stats['row'] = ((df_stats['index'] % num_wells) % settings['rows'])+1
                df_stats['column'] = ((df_stats['index'] % num_wells) // settings['rows'])+1

                ''' requires corrections
                else:
                    if (rows >1):
                        y1 = min(df_stats['y0'])
                        y2 = max(df_stats['y0'])
                        dy = (y2-y1)/(rows - 1)
                        df_stats['row'] = round((df_stats['y0']-y1)/dy) +1
                    if (rows == 1):
                        df_stats['row'] = 1

                    x1 = min(df_stats['x0'])
                    x2 = max(df_stats['x0'])
                    dx = (x2-x1)/(columns - 1)
                    df_stats['index'] = round((df_stats['x0']-x1)/dx)
                '''
    
                df_stats = df_stats.sort_values(by = ["device","row","x0"], ignore_index = None)
                df_stats_clean = df_stats[df_stats['Viability'] != -1]
                cacl_flag = True
                window['-PLOT-'].update(visible = True)

                if settings['hist_flag']:
                    figure, figure_canvas_agg = plot_hist(df_stats_clean, 30, window['-CANVAS-'].TKCanvas, figure_canvas_agg)


                if num_channels >= 3:
                    means = ''
                    for i in range(settings['devices']):
                        line = 'Device ' + str(i+1)+ ': CVmean is '+ str(df_stats_clean[df_stats.device == i+1]['Viability'].mean()) + '\n'
                        means = means + line
                    sg.popup("Cell viability", means)                

                    
        if event == 'Relative cell viability':
            window.close()
            cv2.destroyAllWindows()
            
            fnames = filelist(settings['folder'], ".csv")
            print(fnames)           
            layout_files = [
                [
                    sg.Text("Image Folder"),
                    sg.In(settings['folder'],size=(35, 1), enable_events=True, key="-FOLDER_REPORT-"),
                    sg.FolderBrowse(initial_folder=settings['folder']),
                ],
                [
                    sg.Listbox(values=fnames, enable_events=True, size=(50, 10), key="-FILE LIST_REPORT-")
                ],
                [
                    sg.In(size=(40, 1), key="-FILE1-", enable_events=False),
                    sg.Button('Load day1 report')
                ],
                [
                    sg.In(size=(40, 1), key="-FILE2-", enable_events=False),
                    sg.Button('Load day3 report')                   
                ]]


            layout_buttons = [
                [sg.Button('Report'), 
                sg.Text("From"),
                sg.In('1',size=(7, 1), enable_events=False, key="-FROM-"),
                sg.Text("To"),
                sg.In(settings['columns'],size=(7, 1), enable_events=False, key="-TO-"),
                ],
                [
                    sg.Button('Main menu'),
                    sg.Button('Settings'),
                    sg.Button('Quit')
                ]]
            
            layout_left = [
                    [sg.Text("Day 1")],
                    [sg.Text("")],
                    [sg.Text("Loading: "),
                     sg.In(size=(8,1), key='-LOAD1-')],
                    [sg.Text("CV mean: "),
                     sg.In(size=(8,1), key='-CV1-')],
                    [sg.Text("Live mean: "),
                     sg.In(size=(8,1), key='-LIVE1-')],                
                    [sg.Text("Dead mean: "),
                     sg.In(size=(8,1), key='-DEAD1-')],
                    [sg.Text("")],
                    [sg.Text("")],
                    [sg.Text("Day 3")],
                    [sg.Text("")],
                    [sg.Text("Loading: "),
                     sg.In(size=(8,1), key='-LOAD3-')],
                    [sg.Text("CV mean: "),
                     sg.In(size=(8,1), key='-CV3-')],
                    [sg.Text("Live mean: "),
                     sg.In(size=(8,1), key='-LIVE3-')],                
                    [sg.Text("Dead mean: "),
                     sg.In(size=(8,1), key='-DEAD3-')],
                    [sg.Text("")],
                    [sg.Text("RELATIVE CV: "),
                         sg.In(size=(8,1), key='-CVR-')],

                            ]

            layout_right = layout_files + layout_buttons
                     
            layout = [[sg.Column(layout_right), sg.pin(sg.Column(layout_left, key='-REPORT-', visible = False))]]            
            window = sg.Window('Please choice TWO report files (*.csv)', layout, finalize = True)

                
                
        elif event == "-FOLDER_REPORT-":   # New folder has been chosen
            settings['folder'] = values["-FOLDER_REPORT-"]
            fnames = filelist(settings['folder'], ".csv")
            window["-FILE LIST_REPORT-"].update(fnames)
                
        elif event == "-FOLDER_CORRECT-":   # New folder has been chosen
            settings['folder'] = values["-FOLDER_CORRECT-"]
            fnames = filelist(settings['folder'], ".csv")
            window["-FILE LIST_CORRECT-"].update(fnames)
            
        elif event == "-FILE LIST_CORRECT-":  # A file was chosen from the listbox
            try:
                csv_filename = os.path.join(values["-FOLDER_CORRECT-"], values["-FILE LIST_CORRECT-"][0])
            except:
                pass
                       
            
        elif event == "-FILE LIST_REPORT-":  # A file was chosen from the listbox
            try:
                csv_filename = os.path.join(values["-FOLDER_REPORT-"], values["-FILE LIST_REPORT-"][0])
            except:
                pass
            
        elif event == "Load day1 report":            
            if (csv_filename == ''):
                sg.popup('Cancel',"Please choice the file")
            else:
                window["-FILE1-"].update(csv_filename)

        elif event == "Load day3 report":            
            if (csv_filename == ''):
                sg.popup('Cancel',"Please choice the file")
            else:
                window["-FILE2-"].update(csv_filename)
        
        elif event == "Correct report":
            df_name = csv_filename
            try:
                data = pd.read_csv(df_name)

                for i in range(len(data.index)):
                    data.iat[i,9] = ratio_viability(data.iloc[i]['Int_live'],data.iloc[i]['Int_dead'], live_range = settings['live_range'][cell_type], dead_range = settings['dead_range'][cell_type])        

                data.to_csv(df_name,index=False)
                sg.popup("Report was updated")
            except:
                sg.popup("Error loading file")
            
         
                
                
        
        elif event == "Report":
            d1_name = values["-FILE1-"]
            d3_name = values["-FILE2-"]
            
            w1 = int(values["-FROM-"])
            w2 = int(values["-TO-"])
            
            total_num_wells = settings['rows']*settings['columns']
            
            try:
                df_d1_raw = pd.read_csv(d1_name)
                df_d3_raw = pd.read_csv(d3_name)
                
                df_d1_raw = df_d1_raw.rename(columns={"Viability": "Viability_d1", "Int_live": "Int_live_d1", "Int_dead": "Int_dead_d1", "column": "column_d1"})
                df_d3_raw = df_d3_raw.rename(columns={"Viability": "Viability_d3", "Int_live": "Int_live_d3", "Int_dead": "Int_dead_d3"})
     
                df = pd.concat([df_d1_raw, df_d3_raw], axis=1)
        
                load_d1 = df[df['Viability_d1'] != -1].shape[0]
                load_d3 = df[df['Viability_d3'] != -1].shape[0]
            
                df = df[df['Viability_d1'] != -1]
                df = df[df['Viability_d3'] != -1]
                
                
                df = df[df['column'] >= w1]
                df = df[df['column'] <= w2]
                

                df['CV'] = df['Viability_d3']/ df['Viability_d1']*100
                
                cv = df['CV'].mean()
                
                window["-LOAD1-"].update(load_d1)
                window["-CV1-"].update(round(df['Viability_d1'].mean(),3))
                window["-LIVE1-"].update(round(df['Int_live_d1'].mean(),5))
                window["-DEAD1-"].update(round(df['Int_dead_d1'].mean(),5))
                
                window["-LOAD3-"].update(load_d3)
                window["-CV3-"].update(round(df['Viability_d3'].mean(),3))
                window["-LIVE3-"].update(round(df['Int_live_d3'].mean(),5))
                window["-DEAD3-"].update(round(df['Int_dead_d3'].mean(),5))
                
                window["-CVR-"].update(round(cv,3))
                
                report_flag = True
                
                window['-REPORT-'].update(visible = True)

                if cell_type == 'MCF-7':
                    paste_to_clipboard([total_num_wells - load_d1,load_d1/total_num_wells,round(df['Viability_d1'].mean(),3),'','','','','','', '',total_num_wells - load_d3, 
                                    load_d3/total_num_wells,round(df['Viability_d3'].mean(),3), round(cv,3), '', round(df['Int_live_d3'].mean(),5), round(df['Int_dead_d3'].mean(),5)])
                if cell_type == 'DCBXTO.58':
                    paste_to_clipboard([total_num_wells - load_d1,load_d1/total_num_wells,round(df['Viability_d1'].mean(),3),'','','','','','', '', total_num_wells - load_d3, 
                                    load_d3/total_num_wells,round(df['Viability_d3'].mean(),3), round(cv,3), '', round(df['Int_live_d3'].mean(),5), round(df['Int_dead_d3'].mean(),5)])

            except:
                sg.popup('Cancel',"Error loading files")
                

        if event == 'Synergy':
            window.close()

            layout_files = [
                [
                    sg.Text("Data file"),
                    sg.In(size=(50, 1), enable_events=True, key="-DATAFILE-"),
                    sg.FileBrowse(initial_folder=settings['folder'], size=(8, 1), file_types=(("CSV files", "*.csv"),)),
                ],
                [
                    sg.Text("Fitting file"),
                    sg.In(size=(50, 1), enable_events=True, key="-FITFILE-"),
                    sg.FileBrowse(initial_folder=settings['folder'], size=(8, 1), file_types=(("JSON files", "*.json"),)),
                ]]


            layout_buttons = [
                [sg.Text("CV threshold"),
                 sg.In(settings['cv_threshold'],size=(5, 1), enable_events=True, key="-CV_THRESH-"),
                 sg.Button('Update dataset'),
                ],
                [
                    sg.Button('Main menu'),
                    sg.Button('Settings'),
                    sg.Button('Quit')
                ]]
                 
            layout_left = [
                    [sg.Canvas(key='-CANVAS_SYN-',size=(30, 30))], 
                    [sg.Radio('Synergy', 1,enable_events=True, key = "-SYN_3D-", default=True),
                     sg.Radio('CV', 1,enable_events=True, key = "-CV_3D-"),
                     sg.Radio('Total concentration', 1,enable_events=True, key = "-CONC_3D-"),
                     sg.Radio('Feasibility', 1,enable_events=True, key = "-FEAS_3D-"),
                     sg.Text('Transparency',size=(14,1)),
                     sg.In(settings['3D_alpha'], size=(5, 1), enable_events=True, key="-ALPHA_3D-"),
                     sg.Checkbox('Text', default=settings['3D_text'],enable_events=True, key="-TEXT_3D-")],
                     [sg.Button('Table'),
                     sg.Button('Save 3D plot'), 
                     sg.Button('Save dataset'),
                     sg.Button('Correct fit file'),
                     ],     
                    ]

            layout_right = layout_files + layout_buttons
                     
            layout = [[sg.Column(layout_right), sg.pin(sg.Column(layout_left, key='-SYN-', visible = False))]]            
            window = sg.Window('Please choice data file (*.csv) and fitting file (*.json)', layout, finalize = True)                
            
            if (data_name != '') and (fit_name != ''):
                window["-DATAFILE-"].update(data_name)
                window["-FITFILE-"].update(fit_name)
                figure, figure_canvas_agg = plot_data(data, settings, window['-CANVAS_SYN-'].TKCanvas, figure_canvas_agg)
                window['-SYN-'].update(visible = True)
                window['-SYN_3D-'].update(settings['synergy_flag'])
                window['-CV_3D-'].update(settings['cv_flag'])
                window['-CONC_3D-'].update(settings['conc_flag'])
                window['-FEAS_3D-'].update(not ((settings['synergy_flag'] or settings['cv_flag']) or settings['conc_flag']))
                

                        
    

        if event == 'Update dataset':
            
            data_name = values["-DATAFILE-"]
            fit_name = values["-FITFILE-"]
            settings['cv_threshold'] = values["-CV_THRESH-"]
            settings['synergy_flag'] = values["-SYN_3D-"]
            settings['cv_flag'] = values["-CV_3D-"]
            settings['conc_flag'] = values["-CONC_3D-"]
            
            if (data_name =='') and (fit_name == ''):
                sg.popup('Cancel',"Please choice the files")
            else:
                try:   
                    with open(fit_name) as f:
                        fit_dict = json.load(f)
  
                    data_raw = pd.read_csv(data_name)

                    #invertion of the concentrations and estimation of cell_viability
                    data = norm_data_inverse(data_raw,fit_dict)
                    data['cv_theor'] = cell_viability(data['conc0_inv'], data['conc1_inv'], data['conc2_inv'], fit_dict['fit'][0], fit_dict['fit'][1], fit_dict['fit'][2])

                    #feasibility threashold
                    thresh = float(settings['cv_threshold'])
                    
                    data['feas'] = data['cv_theor'] > thresh
                    data['conc_total'] = conc_total(data['conc0'], data['conc1'], data['conc2'])
                    
                    
                    #plotting of the data

                    figure, figure_canvas_agg = plot_data(data, settings, window['-CANVAS_SYN-'].TKCanvas, figure_canvas_agg)
                    
                    window['-SYN-'].update(visible = True)
                except:
                    sg.popup('Cancel',"Error")

                    
        if event == "Table":
            window.close()            
            layout  = table_show(data, fit_dict, settings['dil_flag']) 

            window = sg.Window('Synergy results', layout, finalize = True)
        
                    
        if event == '-ALPHA_3D-':
            settings['3D_alpha'] = float(values['-ALPHA_3D-'])
            figure, figure_canvas_agg = plot_data(data, settings, window['-CANVAS_SYN-'].TKCanvas, figure_canvas_agg)
       
        
        if event == "-FEAS_3D-":
            settings['synergy_flag'] = values["-SYN_3D-"]
            settings['cv_flag'] = values["-CV_3D-"]
            settings['conc_flag'] = values["-CONC_3D-"]
            figure, figure_canvas_agg = plot_data(data, settings, window['-CANVAS_SYN-'].TKCanvas, figure_canvas_agg)
            
        if event == "-TEXT_3D-":
            settings['3D_text'] = values["-TEXT_3D-"]
            figure, figure_canvas_agg = plot_data(data, settings, window['-CANVAS_SYN-'].TKCanvas, figure_canvas_agg)
            
        if event == "-CV_3D-":
            settings['cv_flag'] = values["-CV_3D-"]
            settings['synergy_flag'] = values["-SYN_3D-"]
            settings['conc_flag'] = values["-CONC_3D-"]
            figure, figure_canvas_agg = plot_data(data, settings, window['-CANVAS_SYN-'].TKCanvas, figure_canvas_agg)
            
        if event == "-SYN_3D-":
            settings['cv_flag'] = values["-CV_3D-"]
            settings['synergy_flag'] = values["-SYN_3D-"]
            settings['conc_flag'] = values["-CONC_3D-"]
            figure, figure_canvas_agg = plot_data(data, settings, window['-CANVAS_SYN-'].TKCanvas, figure_canvas_agg)
            
        if event == "-CONC_3D-":
            settings['cv_flag'] = values["-CV_3D-"]
            settings['synergy_flag'] = values["-SYN_3D-"]
            settings['conc_flag'] = values["-CONC_3D-"]            
            figure, figure_canvas_agg = plot_data(data, settings, window['-CANVAS_SYN-'].TKCanvas, figure_canvas_agg)
            
        if event == "-DIL-":
            settings['dil_flag'] = values["-DIL-"]
            
            window.close()

            layout  = table_show(data, fit_dict, settings['dil_flag']) 
            window = sg.Window('Synergy results', layout, finalize = True)
            
            
        if event == '-STOCK0-':
            fit_dict['stock'][0] = float(values['-STOCK0-'])
            window.close()
            layout  = table_show(data, fit_dict, settings['dil_flag']) 
            window = sg.Window('Synergy results', layout, finalize = True)

        if event == '-STOCK1-':
            fit_dict['stock'][1] = float(values['-STOCK1-'])
            window.close()
            layout  = table_show(data, fit_dict, settings['dil_flag']) 
            window = sg.Window('Synergy results', layout, finalize = True)
            
        if event == '-STOCK2-':
            fit_dict['stock'][2] = float(values['-STOCK2-'])
            window.close()
            layout  = table_show(data, fit_dict, settings['dil_flag']) 
            window = sg.Window('Synergy results', layout, finalize = True)



        if event == 'Save dataset':
            name = sg.popup_get_file('Save results', save_as=True)
            if not name:
                sg.popup("Cancel", "No filename supplied")
            else:
                try:
                    data.to_csv(name + '.csv',index=True)
                except:
                    sg.popup("Cancel", "Error")
        
        
        if event == 'Save 3D plot':
            name = sg.popup_get_file('Save 3D plot', save_as=True)
            if not name:
                sg.popup("Cancel", "No filename supplied")
            else:
                try:
                    figure.savefig(name +'.png')
                except:
                    sg.popup("Cancel", "Error")
        
        if event == 'Correct fit file':
            window.close()
            headings = ['DOX','CPA', '5-FU']
            header =  [[sg.Text('',size=(15,1))] + [sg.Text(h, size=(10,1)) for h in headings]]
            
            
            buttons = [[sg.Button('Update fit file'), sg.Button('Save fit file'), sg.Button('Open a new fit file')],
                      [sg.Button('Synergy'),  sg.Button('Table'), sg.Button('Quit')]]
            
            rows = [[sg.Text('Fitting parameters',size=(30,1))],
                    [sg.Text('E0',size=(15,1)),
                    sg.In(fit_dict['fit'][0][0], size=(10,1), key = "-F00-"),
                    sg.In(fit_dict['fit'][1][0], size=(10,1), key = "-F10-"),
                    sg.In(fit_dict['fit'][2][0], size=(10,1), key = "-F20-")],
                    [sg.Text('H',size=(15,1)),
                    sg.In(fit_dict['fit'][0][1], size=(10,1), key = "-F01-"),
                    sg.In(fit_dict['fit'][1][1], size=(10,1), key = "-F11-"),
                    sg.In(fit_dict['fit'][2][1], size=(10,1), key = "-F21-")],
                    [sg.Text('EC50',size=(15,1)),
                    sg.In(fit_dict['fit'][0][2], size=(10,1), key = "-F02-"),
                    sg.In(fit_dict['fit'][1][2], size=(10,1), key = "-F12-"),
                    sg.In(fit_dict['fit'][2][2], size=(10,1), key = "-F22-")],
                   
                   ]        
            #row = [sg.Text(round(data.iloc[i][f],2), size=(10,1)) for f in features]


            layout = header + rows + buttons
            
            window = sg.Window('Fit file', layout, finalize = True)


        if event == 'Save fit file':
            pass
            
             
                
if __name__ == "__main__":
    main()
    
