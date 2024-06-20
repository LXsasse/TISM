import numpy as np
import torch
import torch.nn as nn

from tangermeme.deep_lift_shap import deep_lift_shap

from tangermeme.ism import saturation_mutagenesis
import matplotlib.pyplot as plt 
from tangermeme.plot import plot_logo

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
    ''' 
    This function uses captum DeepLift or DeepLiftShap to compute sequence attributions. 
    Note that multiply_by_inputs need to be False to be able to compute hypothetical attributions scores. 
    If multiply_by_inputs is False, this function will return local attribution scores, equivalent to gradients
    However, we found that if multiply_by_inputs is set to False, Captum will produce false Deltas because it will falsely directly sum the multipliers instead of multiplying thoese with delta x first
    '''
    
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
    
    x = torch.Tensor(x)
    
    # make sure that model is not using dropout        
    model.eval()
    
    if device is None:
        device = 'cpu'
    
    X_attr = saturation_mutagenesis(model, x, device=device, start = start, end = end, raw_outputs = True, batch_size=batch_size)
    
    refpred = X_attr[0]
    altpred = X_attr[1]
    
    if isinstance(tracks, int):
        tracks = [tracks]
    if tracks is None:
        tracks = np.arange(np.shape(refpred)[-1], dtype = int)
    
    refpred = refpred[..., tracks].cpu().detach().numpy()
    altpred = altpred[...,tracks].cpu().detach().numpy()
    
    size = list(x.size())
    size += [len(tracks)]
    ismout = np.zeros(size)
    print(np.shape(ismout))
    for i, si in enumerate(x):
        # Determine all possible variants
        isnot = torch.where(si[...,start:end] == 0)
        # Assign difference between original and alternative predictions to the bases. 
        for j in range(len(isnot[0])):
            ismout[i, isnot[0][j], isnot[1][j]+start] = altpred[i,isnot[0][j], isnot[1][j]] - refpred[i]
            
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

    plot_logo(att, ax = ax0)
    
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



    
    
