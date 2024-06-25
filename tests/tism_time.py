import numpy as np
import sys, os
import torch
import torch.nn as nn
import time
from scipy.stats import pearsonr
import matplotlib.pyplot as plt 

from tism.models import YuzuAItac, YuzuAItacDeep
from tism.utils import plot_attribution, ism, deepliftshap, plot_bars, write_table
from tism.torch_grad import correct_multipliers, takegrad

from tangermeme.utils import random_one_hot
from tangermeme.ersatz import substitute

from yuzu import precompute
from yuzu.yuzu_ism import yuzu_ism
from yuzu.naive_ism import naive_ism

# (copied from https://github.com/kundajelab/yuzu/blob/main/tutorials/3.%20Using%20Yuzu%20with%20Your%20Model.ipynb)
'''
Limitations of Yuzu

Unfortunately, iterating over model.children() is conceptually easy but requires that the models are sequential. A major consequence of this is that no multi-input or multi-output networks are supported right now (unless the inputs or outputs can be represented in a single tensor). Also, no networks with multiple paths are supported right now, including residual connections.

A second type of limitation is that no operations can be performed in the forward pass other than iterating through the layers in a sequential manner. This means no flattening, reshaping, cropping, adding operations, referring to the same layer multiple times, etc. Any manipulation of the data must occur within the context of layers.
'''

if __name__ == '__main__':

    device = 'cpu'

    Ns= [1, 10, 20]
    b=4
    input_lengths = [250, 500, 1000]
    
    tracks = 1
    track = 0
    
    input_length = 250
    
    # compare two models:
    # AI-tac with three convolutional and 2 fully connected layers
    # Deeper version of AI-tac with 8 conv layers and 3 fully
    model = YuzuAItac(input_length, tracks = tracks)
    modeldeep = YuzuAItacDeep(input_length, tracks = tracks)
    
    t1 = time.time()
    precomputation = precompute(model, input_length, device = device)
    t2 = time.time()
    print('Precomputation Yuzu for AI-TAC', round(t2-t1,2))
    precomputationdeep = precompute(modeldeep, input_length, device = device)
    t3 = time.time()
    print('Precomputation Yuzu for deep AI-TAC', round(t3-t2,2))
    
    runtimes = []
    for N in Ns:
        
        x = random_one_hot((N, b, input_length), random_state = 1).type(torch.float32)
        print('x.shape', x.shape)
        
        comptimes = []
        
        # compute taylor approximated in silico saturation mutagenesis effects.
        t1 = time.time()
        grad_tism = takegrad(x, model, tracks = None, output = 'tism', device = device, baseline = None)
        t2 = time.time()
        print('TISM: AI-TAC: L', input_length, 'N', N, round(t2-t1,2))
        t1d = time.time()
        grad_tismdeep = takegrad(x, modeldeep, tracks = None, output = 'tism', device = device, baseline = None)
        t2d = time.time()
        print('TISM: AI-TACDeep: L', input_length, 'N', N, round(t2d-t1d,2))
        
        comptimes.append([t2-t1,t2d-t1d])
        
        #Apply Yuzu
        t1 = time.time()
        yuzu_isms = yuzu_ism(model, x.numpy(), precomputation, verbose=False, device = device)
        t2 = time.time()
        print('Yuzu: AI-TAC: L', input_length, 'N', N, round(t2-t1,2))
        
        t1d = time.time()
        yuzu_isms = yuzu_ism(modeldeep, x.numpy(), precomputationdeep, verbose=False, device=device)
        t2d = time.time()
        print('Yuzu: AI-TACDeep: L', input_length, 'N', N, round(t2d-t1d,2))
        
        comptimes.append([t2-t1,t2d-t1d])
        
        # compute ISM effects
        t1 = time.time()
        ism_array = ism(x, model, tracks = None, device = device)
        t2 = time.time()
        print('ISM: AI-TAC: L', input_length, 'N', N, round(t2-t1,2))
        
        t1d = time.time()
        ism_array = ism(x, modeldeep, tracks = None, device = device)
        t2d = time.time()
        print('ISM: AI-TACDeep: L', input_length, 'N', N, round(t2d-t1d,2))
        
        comptimes.append([t2-t1,t2d-t1d])
        runtimes.append(comptimes)
        
        '''
        # Compare ISM to TISM
        print('ISM versus TISM')
        for i in range(np.shape(grad_tism)[0]):
            print(i, pearsonr(grad_tism[i,track].flatten(), ism_array[i,track].flatten())[0])
        
        
        # By default Yuzu returns L2 norm of ism values from all tracks. 
        ism_isms = np.sqrt(np.sum(np.square(ism_array), axis = 1))
        
        # Compare ISM to Yuzu ISM
        print('ISM versus Yuzu')
        for i in range(np.shape(grad_tism)[0]):
            print(i, pearsonr(yuzu_isms[i].flatten(), ism_isms[i].flatten())[0])
    
        '''
        
    fig = plot_bars(runtimes, xticklabels = Ns, ylabel = 'time in sec', color = ['cornflowerblue', 'goldenrod', 'indigo'], labels = ['TISM', 'Yuzu', 'ISM'], title = ['AI-TAC', 'AI-TACDeep'], xlabel = 'N')
    figh = plot_bars(runtimes, xticklabels = Ns, ylabel = 'time in sec', color = ['cornflowerblue', 'goldenrod', 'indigo'], labels = ['TISM', 'Yuzu', 'ISM'], title = ['AI-TAC', 'AI-TACDeep'], horizontal = True, xlabel = 'N')
    
    fig.savefig('../results/Comparison_time_N_'+device.replace(':', '_')+'.jpg', bbox_inches = 'tight', dpi = 300)
    fig.savefig('../results/Comparison_time_N_'+device.replace(':', '_')+'.pdf', bbox_inches = 'tight', dpi = 300)

    figh.savefig('../results/Comparison_time_hor_N_'+device.replace(':', '_')+'.jpg', bbox_inches = 'tight', dpi = 300)
    
    write_table(runtimes, '../results/Comparison_time_N_'+device.replace(':', '_')+'.tsv', rows = Ns, columns = ['TISM', 'Yuzu', 'ISM'], additional = ['AITAC', 'AITACDeep'])

    
    if '--show' in sys.argv:
        plt.show()
        
    
    
    N = 1
    
    runtimes = []
    for input_length in input_lengths:
        
        x = random_one_hot((N, b, input_length), random_state = 1).type(torch.float32)
        print('x.shape', x.shape)
        
        comptimes = []
        
        # compare two models:
        # AI-tac with three convolutional and 2 fully connected layers
        # Deeper version of AI-tac with 8 conv layers and 3 fully
        model = YuzuAItac(input_length, tracks = tracks)
        modeldeep = YuzuAItacDeep(input_length, tracks = tracks)
    
        t1 = time.time()
        precomputation = precompute(model, input_length, device = device)
        t2 = time.time()
        print('Precomputation Yuzu for AI-TAC with L', input_length, round(t2-t1,2))
        precomputationdeep = precompute(modeldeep, input_length, device = device)
        t3 = time.time()
        print('Precomputation Yuzu for deep AI-TAC with L', input_length, round(t3-t2,2))
        
        # compute taylor approximated in silico saturation mutagenesis effects.
        t1 = time.time()
        grad_tism = takegrad(x, model, tracks = None, output = 'tism', device = device, baseline = None)
        t2 = time.time()
        print('TISM: AI-TAC: L', input_length, 'N', N, round(t2-t1,2))
        t1d = time.time()
        grad_tismdeep = takegrad(x, modeldeep, tracks = None, output = 'tism', device = device, baseline = None)
        t2d = time.time()
        print('TISM: AI-TACDeep: L', input_length, 'N', N, round(t2d-t1d,2))
        
        comptimes.append([t2-t1,t2d-t1d])
        
        #Apply Yuzu
        t1 = time.time()
        yuzu_isms = yuzu_ism(model, x.numpy(), precomputation, verbose=False, device = device)
        t2 = time.time()
        print('Yuzu: AI-TAC: L', input_length, 'N', N, round(t2-t1,2))
        
        t1d = time.time()
        yuzu_isms = yuzu_ism(modeldeep, x.numpy(), precomputationdeep, verbose=False, device=device)
        t2d = time.time()
        print('Yuzu: AI-TACDeep: L', input_length, 'N', N, round(t2d-t1d,2))
        
        comptimes.append([t2-t1,t2d-t1d])
        
        # compute ISM effects
        t1 = time.time()
        ism_array = ism(x, model, tracks = None, device = device)
        t2 = time.time()
        print('ISM: AI-TAC: L', input_length, 'N', N, round(t2-t1,2))
        
        t1d = time.time()
        ism_array = ism(x, modeldeep, tracks = None, device = device)
        t2d = time.time()
        print('ISM: AI-TACDeep: L', input_length, 'N', N, round(t2d-t1d,2))
        
        comptimes.append([t2-t1,t2d-t1d])
        runtimes.append(comptimes)
        
        '''
        # Compare ISM to TISM
        print('ISM versus TISM')
        for i in range(np.shape(grad_tism)[0]):
            print(i, pearsonr(grad_tism[i,track].flatten(), ism_array[i,track].flatten())[0])
        
        
        # By default Yuzu returns L2 norm of ism values from all tracks. 
        ism_isms = np.sqrt(np.sum(np.square(ism_array), axis = 1))
        
        # Compare ISM to Yuzu ISM
        print('ISM versus Yuzu')
        for i in range(np.shape(grad_tism)[0]):
            print(i, pearsonr(yuzu_isms[i].flatten(), ism_isms[i].flatten())[0])
    
        '''
        
    fig = plot_bars(runtimes, xticklabels = input_lengths, ylabel = 'time in sec', color = ['cornflowerblue', 'goldenrod', 'indigo'], labels = ['TISM', 'Yuzu', 'ISM'], title = ['AI-TAC', 'AI-TACDeep'], xlabel = 'Seq. length')
    figh = plot_bars(runtimes, xticklabels = input_lengths, ylabel = 'time in sec', color = ['cornflowerblue', 'goldenrod', 'indigo'], labels = ['TISM', 'Yuzu', 'ISM'], title = ['AI-TAC', 'AI-TACDeep'], horizontal = True, xlabel = 'Seq. length')
    
    fig.savefig('../results/Comparison_time_L_'+device.replace(':', '_')+'.jpg', bbox_inches = 'tight', dpi = 300)
    fig.savefig('../results/Comparison_time_L_'+device.replace(':', '_')+'.pdf', bbox_inches = 'tight', dpi = 300)
    
    figh.savefig('../results/Comparison_time_hor_L_'+device.replace(':', '_')+'.jpg', bbox_inches = 'tight', dpi = 300)
    
    write_table(runtimes, '../results/Comparison_time_L_'+device.replace(':', '_')+'.tsv', rows = input_lengths, columns = ['TISM', 'Yuzu', 'ISM'], additional = ['AITAC', 'AITACDeep'])
    
    if '--show' in sys.argv:
        plt.show()

    

    
    
    
