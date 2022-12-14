# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 22:54:55 2022

@author: vipro
"""

import cv2
import numpy as np

from skimage import transform

from skimage.filters import threshold_yen
from skimage.segmentation import clear_border
from skimage.morphology import closing, square

from utils.utils import *



def show_img(image_raw, name, scale):
    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)  
    cv2.moveWindow(name, 40,30)
    image_print = transform.resize(image_raw, (image_raw.shape[0]//scale, image_raw.shape[1]//scale))   
    cv2.imshow(name,image_print)


def nothing(event, x, y, flags, params):
    pass

def automatic_brightness_and_contrast(image, clip_hist_percent=1):

    # Calculate grayscale histogram
    hist,bins = np.histogram(image.ravel(),512,range=[0,1])
    hist_size = len(hist)


    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))
    
    
    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray) *255
    #beta = -minimum_gray * alpha
    beta = 0
    
    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)


def crop(image_raw, x_min, x_max, y_min, y_max):
    image = image_raw[int(x_min):int(x_max), int(y_min):int(y_max)]
    return image




#### defining of the coordinates of the wells using the grid
def grid(cord_1, cord_2, cord_3, cord_4, rows, columns, devices): 
    
    #### caclulation of the distances between the wells in the grid
    dy_rows = (cord_2[1] - cord_1[1])/(rows-1)
    dx_rows = (cord_2[0] - cord_1[0])/(rows-1)

    dy_columns = (cord_3[1] - cord_2[1])/(columns-1)
    dx_columns = (cord_3[0] - cord_2[0])/(columns-1)
    
    #### creation of 2D array of the coordinates of the wells
    wells = np.zeros((columns*devices,rows,2))
    wells[0][0] = cord_1
    wells[0][rows-1] = cord_2
    wells[columns-1][rows-1] = cord_3

    for i in range(columns):
        x = wells[0][0][0] + dx_columns*i
        y = wells[0][0][1] + dy_columns*i
        wells[i][0] = (x,y)

        for j in range(rows):
            x = wells[i][0][0] + dx_rows*j
            y = wells[i][0][1] + dy_rows*j
            wells[i][j] = (x,y)
    
    
    if (devices > 1):
        dy_devices = (cord_4[1] - cord_3[1])/(devices-1)
        dx_devices = (cord_4[0] - cord_3[0])/(devices-1)
        
        for k in range(devices-1):
                for i in range(columns):
                    x = wells[0][0][0] + dx_columns*i + dx_devices*(k+1)
                    y = wells[0][0][1] + dy_columns*i + dy_devices*(k+1)
                    wells[i+columns*(k+1)][0] = (x,y)

                    for j in range(rows):
                        x = wells[i+columns*(k+1)][0][0] + dx_rows*j 
                        y = wells[i+columns*(k+1)][0][1] + dy_rows*j
                        wells[i+columns*(k+1)][j] = (x,y)
            
    
    #### reshaping of 2D array to the list of the coordinates
    cords = np.reshape(wells,(-1,2))  
    cords.tolist()
    return cords

def grid_1ch(cord_1, cord_2, cord_3, columns, devices): 
    
    #### caclulation of the distances between the wells in the grid

    dy_columns = (cord_2[1] - cord_1[1])/(columns-1)
    dx_columns = (cord_2[0] - cord_1[0])/(columns-1)
    
    #### creation of 2D array of the coordinates of the wells
    wells = np.zeros((columns*devices,1,2))
    wells[0][0] = cord_1
    wells[columns-1][0] = cord_2

    for i in range(columns):
        x = wells[0][0][0] + dx_columns*i
        y = wells[0][0][1] + dy_columns*i
        wells[i][0] = (x,y)
        
    if (devices > 1):
        dy_devices = (cord_3[1] - cord_2[1])/(devices-1)
        dx_devices = (cord_3[0] - cord_2[0])/(devices-1)
        
        for k in range(devices-1):
            for i in range(columns):
                x = wells[0][0][0] + dx_columns*i + dx_devices*(k+1)
                y = wells[0][0][1] + dy_columns*i + dy_devices*(k+1)
                wells[i+columns*(k+1)][0] = (x,y)

    #### reshaping of 2D array to the list of the coordinates
    cords = np.reshape(wells,(-1,2))  
    cords.tolist()
    return cords





#### circular mask for intensity measurements of each well
def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center >= radius
    return mask


#### calculation of the fluorescence intensities of the wells
def calc_fluo(regions, image, image_live, image_dead, num_channels, settings, device_type, cell_type, bg):
    
    radius = settings['radius'][device_type]
    offset = settings['offset']
    
    radius_mask = radius + offset
    stats = []
    frames =[]
    

    
    if num_channels == 1:

        masked_img = image.copy()    

        thresh = threshold_yen(masked_img)

        for i, region in enumerate(regions):

            #### mask image inside circle
            x0, y0 = region

            # crop image to circle of interest
            x_low = int(round(x0, 0)) - radius_mask
            x_high = int(round(x0, 0)) + radius_mask
            y_low = int(round(y0, 0)) - radius_mask
            y_high = int(round(y0, 0)) + radius_mask

            cropped_img = masked_img[y_low:y_high, x_low:x_high]
            frames.append((cropped_img))

            mask = create_circular_mask(w=radius_mask*2, h=radius_mask*2, center=(radius_mask, radius_mask), radius=radius)

            total_num_pixels = np.sum(~mask)
            
            cropped_img[mask] = 0
            # count how many pixels in mask/circle
            intensity_gray = np.sum(cropped_img) / total_num_pixels
            
            bw = closing(cropped_img < thresh, square(2))
            cleared = clear_border(bw)

            tot_num_pixels = np.sum(~bw)
            area = (total_num_pixels - tot_num_pixels)/total_num_pixels

            stats.append((x0, y0, area, intensity_gray))
    
    
    
    if num_channels == 2:

        masked_img = image.copy()    
        masked_img_live = image_live.copy()

        thresh = threshold_yen(masked_img)

        for i, region in enumerate(regions):

            #### mask image inside circle
            x0, y0 = region

            # crop image to circle of interest
            x_low = int(round(x0, 0)) - radius_mask
            x_high = int(round(x0, 0)) + radius_mask
            y_low = int(round(y0, 0)) - radius_mask
            y_high = int(round(y0, 0)) + radius_mask

            cropped_img = masked_img[y_low:y_high, x_low:x_high]
            cropped_img_live = masked_img_live[y_low:y_high, x_low:x_high]
                        
            frames.append((cropped_img, cropped_img_live))

            mask = create_circular_mask(w=radius_mask*2, h=radius_mask*2, center=(radius_mask, radius_mask), radius=radius)

            total_num_pixels = np.sum(~mask)

            cropped_img[mask] = 0
            cropped_img_live[mask] = 0

            #### calculate sum intensity
            #intensity = np.sum(masked_img) / num_active_pixels  # scale just to get lower number to look at
            intensity_gray = np.sum(cropped_img) / total_num_pixels
            intensity_live = np.sum(cropped_img_live) / total_num_pixels

            # count how many pixels in mask/circle

            bw = closing(cropped_img < thresh, square(2))
            cleared = clear_border(bw)

            tot_num_pixels = np.sum(~bw)
            area = (total_num_pixels - tot_num_pixels)/total_num_pixels

        
            stats.append((x0, y0, area, intensity_gray, intensity_live))


    
    
    if num_channels >= 3:

        masked_img = image.copy()    
        masked_img_live = image_live.copy()
        masked_img_dead = image_dead.copy()

        thresh = threshold_yen(masked_img)

        for i, region in enumerate(regions):

            #### mask image inside circle
            x0, y0 = region

            # crop image to circle of interest
            x_low = int(round(x0, 0)) - radius_mask
            x_high = int(round(x0, 0)) + radius_mask
            y_low = int(round(y0, 0)) - radius_mask
            y_high = int(round(y0, 0)) + radius_mask

            cropped_img = masked_img[y_low:y_high, x_low:x_high]
            cropped_img_live = masked_img_live[y_low:y_high, x_low:x_high]
            cropped_img_dead = masked_img_dead[y_low:y_high, x_low:x_high]
            frames.append((cropped_img, cropped_img_live, cropped_img_dead))

            mask = create_circular_mask(w=radius_mask*2, h=radius_mask*2, center=(radius_mask, radius_mask), radius=radius)

            total_num_pixels = np.sum(~mask)

            cropped_img[mask] = 0
            cropped_img_live[mask] = 0
            cropped_img_dead[mask] = 0

            #### calculate sum intensity
            #intensity = np.sum(masked_img) / num_active_pixels  # scale just to get lower number to look at
            intensity_gray = np.sum(cropped_img) / total_num_pixels
            intensity_live = np.sum(cropped_img_live) / total_num_pixels
            intensity_dead = np.sum(cropped_img_dead) / total_num_pixels 

            # count how many pixels in mask/circle

            bw = closing(cropped_img < thresh, square(2))
            cleared = clear_border(bw)

            tot_num_pixels = np.sum(~bw)
            area = (total_num_pixels - tot_num_pixels)/total_num_pixels
            
            if bg:
                stats.append((x0, y0, area, intensity_gray, intensity_live, intensity_dead))
            else:  
                stats.append((x0, y0, area, intensity_gray, intensity_live, intensity_dead, intensity_live/intensity_dead, ratio_viability(intensity_live,intensity_dead, live_range = settings['live_range'][cell_type], dead_range = settings['dead_range'][cell_type])))

    return stats, frames

def main():
    pass

if __name__ == "__main__":
    main()