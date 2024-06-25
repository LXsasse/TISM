import numpy as np
import torch
import torch.nn as nn
import sys, os

from tangermeme.deep_lift_shap import deep_lift_shap
import logomaker as lm
import pandas as pd
import matplotlib.pyplot as plt 

def deepliftshap(
    x,
    model,
    tracks = None,
    deepshap = False,
    baseline = None,
    device = None,
    batchsize = None,
    multiply_by_inputs = False
    ):
    
    """ 
    This function uses tangermeme's implementation of deepliftshap
    after we realized that captum's implementation does not support GELU
    and other nonlinear functions. 
    Tangermeme also warns you when delta's are large, indicating that 
    your model cannot be used with it's function
    """
    
    if device is None:
        device = 'cpu'
    # set model to use device
    model.to(device)
    
    x = torch.Tensor(x)
    
    if isinstance(tracks, int):
        tracks = [tracks]
    if tracks is None:
        tracks = [0]
    
    
    # batch size will be taken from model
    if batchsize is None:
        batchsize = max(min(x.size(0),int(x.size(0)/x.size(-1)/100000)),1)
    
    if baseline is None:
        basefreq = 1./x.size(-2)
        baseline = torch.ones_like(x)*basefreq
    
    if baseline is not None:
        if baseline.size() != x.size():
            # always need to expand baseline to number of data points to 
            baseline = baseline.unsqueeze(0).expand((x.size(0),) + tuple(baseline.size()))
            # if only frequencies were given, need to expand it along the length of the sequences
            if baseline.size() != x.size():
                baseline = baseline.unsqueeze(-1).expand(tuple(baseline.size())+(x.size(-1),))

    grad = []
    for t, tr in enumerate(tracks):
        gr = deep_lift_shap(model, x, target = tr, references = baseline.unsqueeze(1), device=device, raw_outputs = True)
        grad.append(np.mean(gr.cpu().detach().numpy(), axis = 1))
    
    grad = np.array(grad)
    # return array if shape = (Ndatapoints, Ntracks, Nbase, Lseq)
    grad = np.transpose(grad, axes = (1,0,2,3))
        
    return grad

# Function to compute in silico mutatgenesis effects
def ism(x, 
        model, 
        tracks, 
        device = None,
        start = 0,
        end = -1,
        batch_size = 512,
        ):
    '''
    This function performs three mutagenesis experiments for each location to determine what would happen if the base at that position was mutated into one of the other three bases. 
    '''
    
    x = torch.Tensor(x).to(device)
    
    # make sure that model is not using dropout        
    model.eval()
    
        # set model to use device
    model.to(device)
    
    # prediction for original sequence
    refpred = model.forward(x)
    refpred = refpred.detach().cpu().numpy()
    
    # Some models have distinct output heads, for different species or different modalities.
    if isinstance(refpred, list):
        refpred = np.concatenate(refpred, axis = 1)
    
    if isinstance(tracks, int):
        tracks = [tracks]
    if tracks is None:
        tracks = np.arange(np.shape(refpred)[-1], dtype = int)
    refpred = refpred[..., tracks]
    
    size = list(x.size())
    size += [len(tracks)]
    ismout = np.zeros(size)
    
    for i, si in enumerate(x):
        # Determine all possible variants
        isnot = list(torch.where(si[...,start:end] == 0))
        isnot[-1] += start
        # Generate sequences with variants
        # Needs cloning, otherwise no changes can be made to si
        xalt = torch.clone(si.expand([len(isnot[0])] + list(si.size())))
        for j in range(len(isnot[0])):
            xalt[j,:,isnot[1][j]] = 0
            xalt[j,isnot[0][j],isnot[1][j]] = 1
        # Predict activity of these alternative sequences
        altpred = []
        for j in range(0, xalt.shape[0], batch_size):
            altpr = model.forward(xalt[j:j+batch_size])
            altpr = altpr.detach().cpu().numpy()
            if isinstance(altpr,list):
                altpr = np.concatenate(altpr,axis =-1)
            altpred.append(altpr)
        
        altpred = np.concatenate(altpred, axis = 0)
        altpred = altpred[...,tracks]
        # Assign difference between original and alternative predictions to the bases. 
        for j in range(len(isnot[0])):
            ismout[i, isnot[0][j], isnot[1][j]] = altpred[j] - refpred[i]
            
    ismout = np.swapaxes(np.swapaxes(ismout, 1, -1), -1,-2)
    return ismout

# plot attribution maps with logomaker and plot individual values as heatmap below. 
def plot_attribution(att, # attribution map
                     heatmap = None, # heatmap below
                     figscale = 0.15, 
                     ylabel = None, 
                     ylim = None, 
                     alphabet = 'ACGT'):
    
    # determine limits for attribution plot
    mina = min(0,np.amin(np.sum(np.ma.masked_greater(att,0), axis = -2)))
    maxa = np.amax(np.sum(np.ma.masked_less(att,0), axis = -2))
    attlim = [mina, maxa]
    
    hasheat = heatmap is not None
    if hasheat:
        if not isinstance(heatmap, np.ndarray):
            if heatmap == 'use_attribution':
                heatmap = np.copy(att)
    
    fig = plt.figure(figsize = (figscale*np.shape(att)[1], (10+5*int(hasheat))*figscale), dpi = 50)
    ax0 =  fig.add_subplot(1+int(hasheat), 1, 1)
    ax0.set_position([0.1,0.1+(5*int(hasheat)/(10+5*int(hasheat)))*0.8,0.8,0.8*(10/(10+5*int(hasheat)))])
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.tick_params(bottom = not(hasheat), labelbottom = not(hasheat))
    ax0.grid()

    # create pandas dataframe to provide to logomaker
    alphabet = list(alphabet)
    datadict = {}
    for i in range(len(alphabet)):
        datadict[alphabet[i]] = att[i]
    att = pd.DataFrame(datadict)
    lm.Logo(att, ax = ax0)
    
    if ylabel is not None:
        ax0.set_ylabel(ylabel)
    if ylim is not None:
        ax0.set_ylim(ylim)
    else:
        ax0.set_ylim(attlim)

    if hasheat:
        vlim = np.amax(np.absolute(heatmap))
        ax1 =fig.add_subplot(212)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ta_ = ax1.imshow(heatmap, aspect = 'auto', cmap = 'coolwarm', vmin = -vlim, vmax = vlim)
        ax1.set_yticks(np.arange(len(alphabet)))
        ax1.set_yticklabels(alphabet)
        ax1.set_position([0.1,0.1,0.8,0.8*(4/15)])
    
        axc =fig.add_subplot(991)
        axc.imshow(np.linspace(0,1,101).reshape(-1,1), aspect = 'auto', cmap = 'coolwarm', vmin = 0, vmax = 1)
        axc.set_position([0.9+0.25/np.shape(heatmap)[1],0.1,1/np.shape(heatmap)[1],0.8*(4/15)])
        axc.set_yticks([0,100])
        axc.set_yticklabels([-round(vlim,2), round(vlim,2)])
        axc.tick_params(bottom = False, labelbottom = False, labelleft = False, left = False, labelright = True, right = True)
    
    return fig

def plot_bars(x, width = 0.8, xticklabels=None, xlabel = None, ylabel=None, ylim=None, color = None, figsize = (3.5,3.5), labels = None, title = None, horizontal = False):
    
    """
    Parameter
    ---------
    x : list or np.array, shape = (n_bars,) or (n_bars, n_models), or 
    (n_bars, n_models, n_conditions)
    
    Return
    ------
    
    fig : matplotlib.pyplot.fig object
    
    """

    x = np.array(x)
    positions = np.arange(np.shape(x)[0])
    xticks = np.copy(positions)
    
    if len(np.shape(x)) >1:
        n_models = np.shape(x)[1]
        bst = -width/2
        width = width/n_models
        shifts = [bst + width/2 + width *n for n in range(n_models)]
        positions = [positions+shift for shift in shifts]
        if color is None:
            color = [None for i in range(np.shape(x)[1])]
   
    if len(np.shape(x)) > 2:
        if horizontal: 
            fig = plt.figure(figsize = (figsize[0]* np.shape(x)[-1], figsize[1]))
        else:
            fig = plt.figure(figsize = (figsize[0], figsize[1] * np.shape(x)[-1]))
        
        for a in range(np.shape(x)[-1]):
            if horizontal:
                ax = fig.add_subplot(1, np.shape(x)[-1], a+1)
            else:
                ax = fig.add_subplot(np.shape(x)[-1], 1, a+1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
            for p, pos in enumerate(positions):
                if labels is None:
                    plotlabel = None
                else:
                    plotlabel = labels[p]
                ax.bar(pos, x[:,p,a],  width = width, color = color[p], label = plotlabel)
            
            if labels is not None:
                ax.legend()
        
            ax.grid()
            
            if horizontal: 
                if a == 0:
                    if ylabel is not None:
                        ax.set_ylabel(ylabel)    
                if xticklabels is not None:
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xticklabels, rotation = 60)
                if xlabel is not None:
                    ax.set_xlabel(xlabel)
                
            else:
                if a + 1 == np.shape(x)[-1]:
                    if xticklabels is not None:
                        ax.set_xticks(xticks)
                        ax.set_xticklabels(xticklabels, rotation = 60)
                    if xlabel is not None:
                        ax.set_xlabel(xlabel)
                else:
                    ax.tick_params(labelbottom = False)
            
                if ylabel is not None:
                    ax.set_ylabel(ylabel)
            
            if ylim is not None:
                ax.set_ylim(ylim)
            
            if title is not None:
                ax.set_title(title[a])
        
        if horizontal:
            plt.subplots_adjust(wspace=0.2)
        else:
            plt.subplots_adjust(hspace=0.2)
    else:
        fig = plt.figure(figsize = figsize)
        ax = fig.add_subplot(111)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if len(np.shape(x))>1:
            for p, pos in enumerate(positions):
                if labels is None:
                    plotlabel = None
                else:
                    plotlabel = labels[p]
                ax.bar(pos, x[:,p], width = width, color = color[p], label = plotlabel)
        else:
            ax.bar(positions, x, width = width, color = color, label = label)
        if labels is not None:
            ax.legend()
    
        ax.grid()
        
        if xticklabels is not None:
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, rotation = 60)
        
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        
        if ylim is not None:
            ax.set_ylim(ylim)
    
    return fig
    
def write_table(data, outname, rows, columns = None, additional = None):
    data = np.array(data)
    if columns is None:
        columns = []
    if len(np.shape(data)) == 1:
        data = data.reshape(-1,1)
    if len(np.shape(data)) == 2:
        np.savetxt(outname, np.append(np.array(rows).reshape(-1,1) , data, axis = 1), fmt = '%s', header = ' '.join(columns), delimiter = '\t')
    elif len(np.shape(data)) == 3:
        if additional is None:
            additional = np.arange(np.shape(data)[-1], dtype = int)
        for a, add in enumerate(additional):
            np.savetxt(os.path.splitext(outname)[0] + '_' + add + os.path.splitext(outname)[1], np.append(np.array(rows).reshape(-1,1) , data[..., a], axis = 1), fmt = '%s', header = ' '.join(np.array(columns)), delimiter = '\t')
    

    
