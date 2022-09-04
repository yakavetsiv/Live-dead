# Live-dead
Image processing script for calculation of cell viability of 3D spheroids in microfluidic arrays. 

# Installation
```
pip install -r requirements.txt
```
# Image pre-processing
The best performance is assumed for the images of single quadruplet device. For calculation of cell viability two channels (0:live and 1:dead) are required.

# Running the script
```
python live-dead.py
```
Estimation of the absolute cell viability values for single time point:
1.	Analysis of images
2.	Browse - Select the folder with the example file
3.	Select the image
4.	Load image
5.	Grid 

![image](https://user-images.githubusercontent.com/61687224/188294128-f41547ca-cae6-47fb-8912-bdc4f78e01b9.png)

6. Report
7.	Save the results


# Configuration 
The default settings:
```
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
```

The settings could be saved in ```config.json``` and loaded if needed.
