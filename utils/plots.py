# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 22:57:44 2022

@author: vipro
"""

import PySimpleGUI as sg
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from matplotlib import cm
from utils.utils import *


def plot_hist(data, bins, canvas, figure_canvas_agg, bg):
    
    if figure_canvas_agg:
        figure_canvas_agg.get_tk_widget().forget()
        plt.close('all')

    figure, ax = plt.subplots()
    labels = []
    color_scheme = plt.cm.get_cmap('tab10')
    for i, dev in enumerate(data['device'].unique()):
        data[data.device == dev]['Viability'].plot.hist(density=True, ax=ax, bins = bins, color = color_scheme(i), alpha=0.5, rwidth=0.85)
        #if not(bg):
            #data[data.device == dev]['Viability'].plot.kde(ax=ax, legend=True, bw_method=1, color = color_scheme(i))
        labels.append('Device ' + str(i+1))

    ax.set_ylabel('Probability', fontsize = 14)
    ax.set_xlim(0,100)
    ax.set_xlabel('Cell viability', fontsize = 14)
    ax.legend(labels)
    
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    
    return figure, figure_canvas_agg


def plot_data(data, settings, canvas, figure_canvas_agg):
    
    if figure_canvas_agg:
        figure_canvas_agg.get_tk_widget().forget()
        plt.close('all')
    

    fig = plt.figure(figsize = (10,7))
    ax = fig.add_subplot(projection='3d')
    
    parameters = {'xtick.labelsize': 14,
              'ytick.labelsize': 14,
              'font.family':'sans-serif',
              'font.sans-serif':['Arial']}
    plt.rcParams.update(parameters) 
    
    ax = plt.axes(projection='3d')
    fig.patch.set_facecolor('white')
    
    cv = False
    syn = False
    conc = False
    
    if settings['cv_flag']:
        cmap = plt.cm.Greens
        norm = plt.Normalize(vmin=0, vmax=1)
        cv = True
               
    elif settings['synergy_flag']:
        norm = plt.Normalize(vmin=-0.3, vmax=0.3)
        cmap = plt.cm.RdYlGn
        syn = True
    
    elif settings['conc_flag']:
        norm = plt.Normalize(vmin=0, vmax=1.7)
        cmap = plt.cm.Reds
        conc = True
    
    else:
        cmap = plt.cm.Greens
        norm = plt.Normalize(vmin=-0.5, vmax=0.5)
    
    for i in range(len(data.index)):
        x = data.iloc[i]['conc0']
        y = data.iloc[i]['conc1']
        z = data.iloc[i]['conc2']
        
        if settings['3D_text']:
            if not(np.isnan(data.iloc[i]['ci'])):
                ax.text(x, y, z, data.iloc[i]['name'])
        
        if syn:
            if not (np.isnan(data.iloc[i]['ci'])):
                col = data.iloc[i]['ci']
                p = ax.scatter3D(x, y, z, c=cmap(norm(col)), alpha = settings['3D_alpha'], s=150)
        
        elif cv:
            if not (np.isnan(data.iloc[i]['ci'])):
                col = data.iloc[i]['cv_exp']
                p = ax.scatter3D(x, y, z, c=cmap(norm(col)), alpha = settings['3D_alpha'], s=150)
        
        elif conc:
            if not (np.isnan(data.iloc[i]['ci'])):
                col = data.iloc[i]['conc_total']
                p = ax.scatter3D(x, y, z, c=cmap(norm(col)), alpha = settings['3D_alpha'], s=150)
        
                               
        else:
            if data.iloc[i]['feas']:
                color = 'green'
            else:
                color = 'red'            
            if not(np.isnan(data.iloc[i]['ci'])):
                color = 'grey'
            ax.scatter3D(x, y, z, c=color, alpha = settings['3D_alpha'], s=150)
          
        
        

    ax.zaxis._axinfo['juggled'] = (1,2,0)
    ax.yaxis._axinfo['juggled'] = (0,1,2)
    #ax.set_title(title, fontsize=16)
    ax.xaxis._axinfo['label']['space_factor'] = 1

    ax.set_xlabel('DOX     ', fontsize=16,fontweight='bold', labelpad =10)
    ax.set_ylabel('CPA    ', fontsize=16,fontweight='bold', labelpad =10)
    ax.set_zlabel('5-FU    ', fontsize=16,fontweight='bold', labelpad =10)
    ax.set_ylim(0,1)
    ax.set_xlim(0,1)
    ax.set_zlim(0,1)
        
    if syn:
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), shrink=0.6, aspect=20)
        cbar.set_label("CI", fontsize=16, fontweight='bold', labelpad =10)

    
    elif cv:
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), shrink=0.6, aspect=20)
        cbar.set_label("CV", fontsize=16, fontweight='bold', labelpad =10)
        
    elif conc:
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), shrink=0.6, aspect=20)
        cbar.set_label("Total concentration", fontsize=16, fontweight='bold', labelpad =10)

    
    else:
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), shrink=0.6, aspect=20)
    
    figure_canvas_agg = FigureCanvasTkAgg(fig, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    
    return fig, figure_canvas_agg

def table_show(data, fit_dict, dil):
    
    features =  ['conc0_inv','conc1_inv', 'conc2_inv', 'ci','cv_theor']
    headings = ['DOX','CPA', '5-FU', 'CI','CV theor']
    buttons = [[sg.Checkbox('Dilutions', default=dil, key="-DIL-", enable_events=True)],
        [sg.Button('Synergy'),sg.Button('Save dataset'), sg.Button('Correct fit file'),sg.Button('Main menu'),sg.Button('Quit')]]

    if dil:
        data = dil_table(data, fit_dict['stock'])
        features =  ['conc0_inv','conc1_inv', 'conc2_inv', 'ci','cv_theor','vol0','vol1','vol2']
        headings = ['DOX','CPA', '5-FU', 'CI','CV theor','Vol DOX','Vol CPA','Vol 5-FU']
        buttons = [[sg.Text('Stocks', size = (55,1)),sg.In(fit_dict['stock'][0], size = (10,1), enable_events=True, key = '-STOCK0-'),sg.In(fit_dict['stock'][1], size = (10,1), enable_events=True, key = '-STOCK1-'),sg.In(fit_dict['stock'][2], size = (10,1), enable_events=True, key = '-STOCK2-')],
                   [sg.Checkbox('Dilutions', default=dil, key="-DIL-", enable_events=True)], 
                   [sg.Button('Synergy'),sg.Button('Save dataset'), sg.Button('Correct fit file'), sg.Button('Main menu'),sg.Button('Quit')]]

    header =  [[sg.Text(h, size=(10,1)) for h in headings]]

    rows = []
    
    for i in range(len(data)):
        row = [sg.Text(round(data.iloc[i][f],2), size=(10,1)) for f in features]
        rows.append(row)
  
    rows = [[sg.Column(rows, scrollable=True)]]
    layout = header + rows + buttons
    
    return layout

def main():
    pass

if __name__ == "__main__":
    main()