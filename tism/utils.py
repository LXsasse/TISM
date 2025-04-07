import numpy as np
import torch
import torch.nn as nn
import sys, os

from tangermeme.deep_lift_shap import deep_lift_shap
import logomaker as lm
import pandas as pd
import matplotlib.pyplot as plt 
from .torch_grad import correct_multipliers

def deepliftshap(
    x,
    model,
    tracks = None,
    baseline = None,
    device = None,
    batchsize = None,
    multiply_by_inputs = False, 
    output = 'hypothetical'
    ):
    
    """ 
    This function uses tangermeme's implementation of deepliftshap
    after we realized that captum's implementation does not support GELU
    and some other nonlinear functions. 
    Tangermeme also warns you when delta's are large, indicating that 
    your model architecutre cannot be approximated with it's implementation.

    Parameters
    ----------
    x : torch.Tensor
        Input data. Shape = (Ndatapoints, Nbases, Lseq)
    model : torch.nn.Module
        Model to be used for attribution.
    tracks : list of int, or int, 
        List of tracks to be used for attribution. If None, all tracks are 
        used.
    baseline : torch.Tensor or size (Nbases,), (Nbases, Lseq), or (Ndatapoints,
        Nbases, Lseq)
        Baseline to be used for attribution. If None, a baseline of 1./Nbases
        is used.
    device : str or torch.device
        Device to be used for attribution. If None, 'cpu' is used.
    batchsize : int
        Batch size to be used for attribution. If None, batch size is 
        determined from the model and the length of the sequences.
    multiply_by_inputs : bool
        If True, the attribution is multiplied by the input for visualization.
        If False, 'local' or 'global' is used. 'local' means that the multipliers
        are returned for each base, 'global' means that the multipliers are
        multiplied by the differences between the input and the baseline.
    output : str
        Type of output to be used. If 'corrected', the mean of the gradients
        is subtracted from the gradients. If 'hypothetical', the gradients are
        normalized by the difference between the input and the baseline. 
        For uniform baselines, this is the same as corrected. 
        If 'global', the gradients are multiplied by the difference between 
        the input and the baseline. 
        If 'local', the gradients are returned as is.
        If 'tism', the gradients of the base are subtracted from the others.    
    Returns
    """
    
    if device is None:
        device = 'cpu'
    # set model to use device
    model.to(device)
    
    # make sure that the input is a tensor
    if not isinstance(x, torch.Tensor):
        if isinstance(x, np.ndarray):
            x = torch.Tensor(x)
        else:
            raise ValueError('x must be a torch.Tensor or numpy.ndarray')
    if len(x.size()) < 3:
        x = x.unsqueeze(0)
    
    
    if isinstance(tracks, int):
        tracks = [tracks]
    if tracks is None:
        tracks = np.arange(model.forward(x[[0]]).size(-1), dtype = int)
    
    
    # batch size will be taken from model and the length of the
    # sequences.
    if batchsize is None:
        batchsize = max(min(x.size(0),int(x.size(0)/x.size(-1)/100000)),1)
    
    # if baseline is None, use a baseline of 1./Nbases
    if baseline is None:
        basefreq = 1./x.size(-2)
        baseline = torch.ones_like(x)*basefreq
    
    # adjust baseline to match the input size
    if baseline is not None:
        if baseline.size() != x.size():
            # always need to expand baseline to number of data points to 
            baseline = baseline.unsqueeze(0).expand((x.size(0),) + 
                    tuple(baseline.size()))
            # if only frequencies were given, need to expand it along the 
            # length of the sequences
            if baseline.size() != x.size():
                baseline = baseline.unsqueeze(-1).expand(tuple(baseline.size())
                        +(x.size(-1),))

    grad = []
    # compute deeplift all all tracks
    for t, tr in enumerate(tracks):
        # Tangermeme's implementation of deepliftshap computes deeplift to 
        # to multiple baselines, so we need to expand the baseline to match the input
        # size.
        gr = deep_lift_shap(model, x, target = tr, 
                references = baseline.unsqueeze(1), device=device,
                raw_outputs = True)
        # take the mean over all baselines, in our implementation this is
        # is just one. 
        grad.append(np.mean(gr.cpu().detach().numpy(), axis = 1))
    
    grad = np.array(grad)
    # return array if shape = (Ndatapoints, Ntracks, Nbase, Lseq)
    grad = np.transpose(grad, axes = (1,0,2,3))
    # correct the multipliers depending on the output type
    grad = correct_multipliers(grad, output, x = x.detach().cpu().numpy(), baseline = baseline.detach().cpu().numpy(), channel_axis = -2 )
    
    if multiply_by_inputs:
        # multiply the gradients by the inputs
        grad = grad * x.unsqueeze(1)
    return grad

# Function to compute in silico mutatgenesis effects
def ism(x, 
        model, 
        tracks, 
        device = None,
        start = 0,
        end = -1,
        batch_size = 512,
        output = 'ism',
        multiply_by_inputs = False,
        ):
    '''
    This function performs three mutagenesis experiments for each location 
    to determine what would happen if the base at that position was mutated 
    into one of the other three bases.
    x : torch.Tensor or numpy.ndarray
        Input data. Shape = (Ndatapoints, Nbases, Lseq)
    model : torch.nn.Module
        Model to be used for attribution.
    tracks : list of int, or int,
        List of tracks to be used for attribution. If None, all tracks are
        used.
    device : str or torch.device
        Device to be used for attribution. If None, 'cpu' is used.
    start : int
        Start position of the sequence to be mutated. If None, 0 is used.
    end : int
        End position of the sequence to be mutated. If None, -1 is used.
    batch_size : int
        Batch size to be used for computation. If None, batch size is
        determined from the model and the length of the sequences.
    output : str 
        If 'ism', the ism values are returned for each base that was mutated.
        If 'ism_attributions', the mean of each position is subtracted from the ism
        values.
        If 'predictions' the predictions are returned for each base that was mutated.
        If 'mean_effects', the mean effect of each of the three substituions
        is subtracted from the ism
    multiply_by_inputs : bool   
        If True, and output not 'ism' the attributions are multiplied by the input for visualization.
    '''
    
    # set device to cpu if None
    if device is None:
        device = 'cpu'
    
    # make sure that model is not using dropout        
    model.eval()
        # set model to use device
    model.to(device)
    
    # make sure that the input is a tensor
    if not isinstance(x, torch.Tensor):
        if isinstance(x, np.ndarray):
            x = torch.Tensor(x)
        else:
            raise ValueError('x must be a torch.Tensor or numpy.ndarray')
    if len(x.size()) < 3:
        x = x.unsqueeze(0)

    # prediction for original sequence with batchsize
    if batch_size is None:
        batch_size = max(min(x.size(0),int(x.size(0)/x.size(-1)/100000)),1)
    # batch size will be taken from model and the length of the
    # sequences.
    refpred = []
    for i in range(0, x.shape[0], batch_size):
        refpred.append( model.forward(x[i:i+batch_size].to(device)).detach().cpu().numpy())
    refpred = np.concatenate(refpred, axis = 0)
    
    # Some models have distinct output heads, for different species or different
    # modalities.
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
            altpr = model.forward(xalt[j:j+batch_size].to(device))
            altpr = altpr.detach().cpu().numpy()
            if isinstance(altpr,list):
                altpr = np.concatenate(altpr,axis =-1)
            altpred.append(altpr)
        
        altpred = np.concatenate(altpred, axis = 0)
        altpred = altpred[...,tracks]
        # Assign difference between original and alternative predictions to the
        # bases. 
        if output == 'predictions':
            ismout[i][:] = refpred[i]
            for j in range(len(isnot[0])):
                ismout[i, isnot[0][j], isnot[1][j]] = altpred[j]
        else:
            for j in range(len(isnot[0])):
                ismout[i, isnot[0][j], isnot[1][j]] = altpred[j] - refpred[i]
            
    ismout = np.swapaxes(np.swapaxes(ismout, 1, -1), -1,-2)
    
    # Correct the multipliers depending on the output type
    if output == 'ism' or output == 'predictions':
        # no correction needed
        pass
    elif output == 'ism_attributions':
        ismout = ismout - np.mean(ismout, axis = -2, keepdims = True)
    elif output == 'mean_effects':
        ismout = ismout - np.sum(ismout, axis = -2, keepdims = True)/(ismout.shape[-2]-1)

    if multiply_by_inputs and output != 'ism' and output != 'predictions':
        # multiply the gradients by the inputs
        ismout = ismout * x.unsqueeze(1)
    return ismout

# plot attribution maps with logomaker and plot individual values as 
# heatmap below. 
def plot_attribution(att, # attribution map
                     heatmap = None, # heatmap below
                     figscale = 0.15, 
                     lenscale = 1.,
                     ylabel = None, 
                     xticklabels = None,
                     xticks = None,
                     dpi = 50,
                     ylim = None, 
                     grid = True,
                     alphabet = 'ACGT'):
    """
    Parameters
    ----------
    att : np.ndarray, shape = (Nbases, Lseq)
        Attribution map to be plotted.
    heatmap : np.ndarray, shape = (Nbases, Lseq)
        Heatmap to be plotted below the attribution map. If None, no heatmap
        is plotted. If 'use_attribution', the attribution map is used as heatmap.
    figscale : float
        Scale for the individual positions.
    lenscale : float
        Scale for the length to height ratio of the sequence attributions
    ylabel : str
        Label for the y-axis.
    xticklabels : list of str
        Labels for the x-axis. If None, the x-axis is not labeled.
    xticks : list of int
        Ticks for the x-axis. If None, the x-axis is not ticked.
    dpi : int
        Dots per inch for the figure.
    ylim : list of float
        Limits for the y-axis. If None, the y-axis is not limited.
    grid : bool
        If True, a grid is plotted.
    alphabet : str
        Alphabet to be used for the y-axis of the heatmap and the data frame for logomaker
    Returns
    -------
    fig : matplotlib.pyplot.fig object
    """
    # check if att is a numpy array
    if not isinstance(att, np.ndarray):
        if isinstance(att, torch.Tensor):
            att = att.detach().cpu().numpy()
        else:
            raise ValueError('att must be a numpy array or torch tensor')
    
    # determine limits for attribution plot
    mina = min(0,np.amin(np.sum(np.ma.masked_greater(att,0), axis = -2)))
    maxa = np.amax(np.sum(np.ma.masked_less(att,0), axis = -2))
    attlim = [mina, maxa]
    
    hasheat = heatmap is not None
    if hasheat:
        if not isinstance(heatmap, np.ndarray):
            if heatmap == 'use_attribution':
                heatmap = np.copy(att)
    
    fig = plt.figure(figsize = (figscale*np.shape(att)[1]*lenscale, 
        (10+5*int(hasheat))*figscale), dpi = dpi)
    
    ax0 =  fig.add_subplot(1+int(hasheat), 1, 1)
    ax0.set_position([0.1,0.1+(5*int(hasheat)/(10+5*int(hasheat)))*0.8,0.8,
        0.8*(10/(10+5*int(hasheat)))])
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    if xticks is not None:
        ax0.set_xticks(xticks)
        ax0.set_xticklabels(xticks)
    elif xticklabels is not None:
        if xticklabels < att.shape[-1]:
            ax0.set_xticks(xticklabels)
        ax0.set_xticklabels(xticklabels, rotation = 60)
    
    ax0.tick_params(bottom = not(hasheat), labelbottom = not(hasheat))
    if grid:
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
        ta_ = ax1.imshow(heatmap, aspect = 'auto', cmap = 'coolwarm', 
                vmin = -vlim, vmax = vlim)
        ax1.set_yticks(np.arange(len(alphabet)))
        ax1.set_yticklabels(alphabet)
        ax1.set_position([0.1,0.1,0.8,0.8*(4/15)])
        if xticks is not None:
            ax1.set_xticks(xticks)
            ax1.set_xticklabels(xticks)
        elif xticklabels is not None:
            if xticklabels < att.shape[-1]:
                ax1.set_xticks(xticklabels)
            ax1.set_xticklabels(xticklabels, rotation = 60)
    
        axc =fig.add_subplot(991)
        axc.imshow(np.linspace(0,1,101).reshape(-1,1), aspect = 'auto', 
                cmap = 'coolwarm', vmin = 0, vmax = 1, origin = 'lower')
        axc.set_position([0.9+0.25/np.shape(heatmap)[1],0.1,
            1/np.shape(heatmap)[1],0.8*(4/15)])
        axc.set_yticks([0,100])
        axc.set_yticklabels([-round(vlim,2), round(vlim,2)])
        axc.tick_params(bottom = False, labelbottom = False, labelleft = False,
                left = False, labelright = True, right = True)
    
    return fig

def plot_bars(x, width = 0.8, xticklabels=None, xlabel = None, ylabel=None, 
        ylim=None, color = None, figsize = (3.5,3.5), labels = None, 
        title = None, horizontal = False):
    
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
            fig = plt.figure(figsize = (figsize[0]* np.shape(x)[-1], 
                figsize[1]))
        else:
            fig = plt.figure(figsize = (figsize[0], 
                figsize[1] * np.shape(x)[-1]))
        
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
                ax.bar(pos, x[:,p,a],  width = width, color = color[p], 
                        label = plotlabel)
            
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
                ax.bar(pos, x[:,p], width = width, color = color[p], 
                        label = plotlabel)
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
        np.savetxt(outname, np.append(np.array(rows).reshape(-1,1), 
            data, axis = 1), fmt = '%s', header = ' '.join(columns), delimiter = '\t')
    elif len(np.shape(data)) == 3:
        if additional is None:
            additional = np.arange(np.shape(data)[-1], dtype = int)
        for a, add in enumerate(additional):
            np.savetxt(os.path.splitext(outname)[0] + '_' + add + 
                    os.path.splitext(outname)[1], 
                    np.append(np.array(rows).reshape(-1,1), 
                        data[..., a], axis = 1), fmt = '%s', 
                    header = ' '.join(np.array(columns)), delimiter = '\t')
    

    
