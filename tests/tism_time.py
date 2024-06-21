import numpy as np
import sys, os
import torch
import torch.nn as nn
import time
from scipy.stats import pearsonr
import matplotlib.pyplot as plt 

from tism.models import seqtofunc_cnn_yuzu
from tism.tangermeme_utils import plot_attribution, ism, deepliftshap
from tism.torch_grad import correct_multipliers, takegrad

from tangermeme.utils import random_one_hot
from tangermeme.ersatz import substitute

from yuzu import precompute
from yuzu.yuzu_ism import yuzu_ism
from yuzu.naive_ism import naive_ism

if __name__ == '__main__':

    '''
    parameters = '../data/deepsea.beluga.pth'
    model = Beluga()
    model.load_state_dict(torch.load(parameters))
    model.eval()
    '''
    
    N=1
    b=4
    input_length = 200
    
    x = random_one_hot((N, b, input_length), random_state = 1).type(torch.float32)
    x = substitute(x, "CTCAGTGATG")
    #x = x.detach().cpu().numpy()
    print(x.shape)
    model = seqtofunc_cnn_yuzu(b, input_length, n_kernels = 200, l_kernels = 15, l_conv=7, padding = 2/3, N_convs = 3, pooling_size = 3, n_tracks = 1)
    
    #y = model.forward(torch.Tensor(x))
    
    track = 2
    vis_seq = 0
    
    
    
    '''
    # compute taylor approximated in silico saturation mutagenesis effects.
    t1 = time.time()
    grad_tism = takegrad(x, model, tracks = track, output = 'tism', device = None, baseline = None, channel_axis = -2)
    t2 = time.time()
    print("TISM values computed from gradients in", '{}s'.format(round(t2-t1,3)), 'of shape', np.shape(grad_tism))
    
    # TISM can be used just like ISM values, for example plotting the mean effect of mutating a base to determine the importance of that base.
    tism0 = grad_tism[vis_seq,0]
    meaneffect_tism = -np.sum(tism0/3, axis = -2)[None,:] * x[vis_seq]
    fig_tism = plot_attribution(meaneffect_tism[...,900:1100], heatmap = tism0[...,900:1100], ylabel = 'Mean\nTISM')


    
    # compute ISM effects
    t1 = time.time()
    ism_array = ism(x, model, tracks = track, start = 900, end = 1100)
    t2 = time.time()
    print("ISM values would be computed for 2,000bp in", '{}s'.format(round(10*(t2-t1),3)), 'of shape', np.shape(ism_array))
    
    ism0 = ism_array[vis_seq,0]
    meaneffect_ism = -np.sum(ism0/3, axis = -2)[None,:] * x[vis_seq]
    fig_ism = plot_attribution(meaneffect_ism[...,900:1100], heatmap = ism0[...,900:1100], ylabel = 'Mean\nISM')
    
    # Compare ISM to TISM
    print('ISM versus TISM')
    for i in range(np.shape(x)[0]):
        print(i, pearsonr(grad_tism[i,0][...,900:1100].flatten(), ism_array[i,0][...,900:1100].flatten())[0]) #, pearsonr(mean_grad_tism[i,0], mean_grad_ism[i,0])
    '''
    
    
    
    # (copied from https://github.com/kundajelab/yuzu/blob/main/tutorials/3.%20Using%20Yuzu%20with%20Your%20Model.ipynb)
    '''
    Limitations of Yuzu

    Unfortunately, iterating over model.children() is conceptually easy but requires that the models are sequential. A major consequence of this is that no multi-input or multi-output networks are supported right now (unless the inputs or outputs can be represented in a single tensor). Also, no networks with multiple paths are supported right now, including residual connections.

    A second type of limitation is that no operations can be performed in the forward pass other than iterating through the layers in a sequential manner. This means no flattening, reshaping, cropping, adding operations, referring to the same layer multiple times, etc. Any manipulation of the data must occur within the context of layers.
    '''
    
    #Now let's apply Yuzu to the second sequence.    
    precomputation = precompute(model, input_length)

    yuzu_ism_scores = yuzu_ism(model, x, * precomputation, verbose=False)[0]
    
    

    
    
    
