import numpy as np
import torch
import toch.nn as nn



# Function to normalize and correct attributions/values at a loci
def correct_multipliers(grad, # array of values that will be corrected
                        output, # type of correction, see below
                        x = None, # the sequence from which the values were generated
                        baseline = None, # the baseline to which values were computed
                        channel_axis = -2 # the axis of the bases, in my models inputs are of shape (N_samples, n_bases, l_seqs)
                        ):
    if output == 'corrected': # corrected gradients as defined by Majdandzic et al. 2023, https://link.springer.com/article/10.1186/s13059-023-02956-3
        grad = grad - np.mean(grad, axis = channel_axis)[..., None, :]
    elif output == 'global': # global attributions, see "A unified view on gradient attribution methods for deep neural networks", Ancona et al. 2017 DOI:10.3929/ETHZ-B-000237705
        grad = grad * (x- baseline)[:,None]
    elif output == 'hypothetical': # hypothetical attributions are a third type of correction that give a sense of what importance would be placed on a different base in the sequence if that base were present. See here: https://github.com/kundajelab/tfmodisco/issues/5
        # They are important for motif detection and clustering in TFmodisco https://arxiv.org/abs/1811.00416
        grad = grad - np.expand_dims(np.sum(baseline * grad, axis = channel_axis), channel_axis)
    elif 'ISM' in output.upper():
        # returns effects for every base-pair substitution, i.e apprximation of f(x0)-f(x1), for which x0 and x1 differ between one base pair.
        grad = grad - np.expand_dims(np.sum(grad * x[:,None], axis = channel_axis), channel_axis)
    return grad
    

def takegrad(x, 
             model, 
             tracks = None, 
             output = 'corrected', 
             device = None, 
             baseline = None,
             channel_axis = -2):
    '''
    Input:
    x       :   input one-hot encoded sequence of shape=(N_data_points, N_bases, Length), can be torch.tensor or numpy array, x is transformed into torch tensor to be given to the model.
    model   :   pytorch model with forward function
    tracks  :   columns in predicted output that are of interest, None will return attributions for all columns
    output  :   Determines normalization of gradient, can be 'local', 'global', 'corrected', 'hypothetical', or 'tism'
    device  :   Define device to compute gradients on, None should be 'cpu'
    baseline:   either baseline frequencies for bases shape=(N_bases,), singular one-hot encoded sequence or frequencies shape=(N_bases, Length), or set of one-hot encoded sequence or frequencies with the same shape as x, shape=(N_data_points, N_bases, Length)
    
    Returns: 
    grad    :   np.ndarray of shape (N_data_points, N_tracks, N_bases, Length)
    
    # Missing: 
    - Automatic batches
    - More memory efficient way of dealing with generation of baselines
    
    '''
    
    # Independent of the data type of x, convert x to torch.Tensor
    x = torch.Tensor(x)
    
    if channel_axis == -1:
        seqlen_axis = -2
    else:
        seqlen_axis = -1
    
    # set to uniform by default if None and required by output
    if baseline is None:
        if output == 'global' or output == 'hypothetical':
            baseline = np.ones(x.size())/x.size(channel_axis)
    
    # convert baseline to numpy array
    if isinstance(baseline, torch.Tensor):
        baseline = baseline.numpy()
    
    # baseline can be basepair frequencies, single sequence of frequencies, or set of sequences
    if baseline is not None:
        if np.shape(baseline) != x.size():
            # always need to expand baseline to number of data points to 
            baseline = np.repeat(np.expand_dims(baseline, 0), x.size(0), axis = 0)
            # if only frequencies were given, need to expand it along the length of the sequences
            if np.shape(baseline) != x.size():
                baseline = np.repeat(np.expand_dims(baseline, seqlen_axis), x.size(seqlen_axis), axis = seqlen_axis)
            
    
    if isinstance(tracks, int):
        tracks = [tracks]
    
    # make sure that model is not using dropout        
    model.eval()
    # set model to use device
    model.to(device)
    
    # collect gradients in here
    grad = []
    for i in range(x.size(dim = 0)):
        # I don't remember why I needed this but there was an issue if I applied backward multiple times to the same object
        xi = torch.clone(x[[i]])
        xi.requires_grad = True
        # collects gradients over selected tracks
        grad_per_track = []
        pred = model.forward(xi)
        # My model can return list of tensors, e.g. if you want to predict different data modalities
        if isinstance(pred, list):
            pred = torch.cat(pred, axis = 1)
        if tracks is None: # use all output tracks as default
            tracks = torch.arange(pred.size(dim = -1), dtype = int)
        
        # iterate over the selected tracks (columns of output matrix) and collect gradients
        for t, tr in enumerate(tracks):
            pred[0, tr].backward(retain_graph = True)
            gr = xi.grad.clone().cpu().numpy()
            grad_per_track.append(gr)
            xi.grad.zero_()
        grad.append(np.concatenate(grad_per_track,axis = 0))
    grad = np.array(grad)
    
    # gradient outputs are by default 'local' attributions, aka multipliers
    # adjust local attributions to user definition
    grad = correct_multipliers(grad, output, x.numpy(), baseline, channel_axis = channel_axis)
        
    return grad
