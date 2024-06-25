import numpy as np
import sys, os
import torch
import torch.nn as nn
import time
from scipy.stats import pearsonr
import matplotlib.pyplot as plt 

from tism.models import seqtofunc_cnn
from tism.utils import plot_attribution, ism, deepliftshap
from tism.torch_grad import correct_multipliers, takegrad

from tangermeme.utils import one_hot_encode
from tangermeme.ersatz import substitute

if __name__ == '__main__':
    
        # model parameter pth file 
    parameters = '../data/Quickfitmodel_parameter.pth'
    
    # This model was trained on ATAC-seq data from https://www.pnas.org/doi/10.1073/pnas.2011795117
    # bed files and ATAC-seq data can be downloaded from here:
    # https://github.com/smaslova/AI-TAC/
    
    # Know your model's length of input sequences
    input_length = 251
    # DNA sequence with A,C,G,T
    b = 4
    
    model = seqtofunc_cnn(b, input_length, n_kernels = 200, N_convs = 3, pooling_size = 3, n_tracks = 81)
    params = torch.load(parameters, map_location=torch.device('cpu'))
    model.load_state_dict(params)
    model.eval()
    
    
    # Sequences with signal
    seqs = [ 'TAAAACAGTTTACAGTAGGCTGGTAAGAGCCTGCAGCTGGAGGGTTCAGGGTCTTCCAATTATGCAAGTCTACTCTTCCCCAGCGTATGACTTTGAGAATTACTTCCTCTTTCCACAGGCTCAACAAGCTACTCAGAGCAGGTCAAGGAGGGAACAGAGAGTCAACTGCTTCCTCTTCCTGCTGTTGGCTGAGATGATAGGTGAAAGTTTTTAATTTCTCTAGGTGAGTTGATTAAGCCTAAAAGATTTCC', 'TCCTCCTGGACTGTAATGGCCTAGGGTGAAAGTTTAGTGTCTCATGGGAGTCTGGTCTTGAGCAATCAGGCTGGACCCCAAGGGTAAGAAGCCTCCACTCTTCCCTTGAGACCTTGTGGTCTCTTGTGGACTTTGAAGAAAGGATATTTTCATCCACCCATCCCTTGAGGTCGCAGAATCCTGTGTTTAGGGCTTGGCATCTGTCCTGGGTGATCTCAGGCCTTCACCTCCTGCTCTCACTGCAGAGCCTG',
    'TGGATTTTGATCGAGTTAATTTTAAAAAGACACTTTAGTTTGCTGTGCTAATTAATATTTTCATTAAAGTACTGCATAGTATTGCATAGACATTTGTTCTGACCCATTTAAAACCTCACATTCTCGTATTCTCTGTCCTGTGACCTACATAAAGTCGTAGGTAATGTTCTTAATATTTAGAGCTTTTGTAGTCTTGTACACAATAAGAGGTGTAACCAGGACTGTAATAAAAAGTTTCCTTAAAACAATAA',
    'GCTGTACCCAAGTCCTCTTCAAGAGCAGTACACAACTCTTAATGACCAGGTCACCTCTCCAGCCCTCCAATATCTTTCTTTGTGATTGAATCTAGGTCATTGTTCTTCCTGTGCGGCTTTTGGGTTTCTGTGTATGAGCTCAGCTAGCTCCTCTAACCAACAGGACCCAAGTAGGACAGCCCCAGGGACTGAGTTCTCAGCACTCTGGTTTTCAAACTACATGTGCACAAGCGTCCTTTGTTTTGTCCGGA',
    'AAATACAACCATCCATTGCTTAGTGACAGGGGCAAACTTTGAGGAATTTGTTTCTGGGCAATGTTGTTGTTAGGCAAGCACCATAACGTATATTCACACAACAGAGTTAACTAGGTCATCACTAAGAGAAATTGTCCTGAGAGACAGCAATTGTCCGTGTGGTCTACTGCAGACCACAATGTCATTACATCATATATTACCTATTGTCAAGTTCTCTGTTTGATAGTTTGTTTTATCCTTAGATAATAAAT',
    'TCACAAGCCCTATCTAACAACACCAGCTGGGCCTTCACCCATAATAGGACTGCATGTATGGTAGTTTTAACTCTGACTCTCAGAAGTCAGACTGACCTACATGTACATCTGCCTTGAGGCCTTAACCACCGTTCAGTATTGGTAATTAACACTGGAAACCAATACAGGATTCTCTTTGTGCTCTTGGTGACTTTAAGTAGGTCATCTTAGCATCACCACTTTGAAGCTGGGACATACTTGACCTAATTTTT',
    'GCTCTGGGACAAGGACAGCATTGAAGAAGCTGGGCTGATTCTCACTGCTTGGCCTTGAGTGGGACCCCTTGAGCTCCCCAGAGTCCCTAGTCTGTGGATTTAGTGGTTTTAACACATGGGCAGACATCTGGGCTTCATCAAACCACTATACCCATGGGGACATGTTCTCCCTTGCCGAACCATTCAGAAGAGCCCTAGGGAACCAGAGACATCTCCCTGCAGAGGGGGTGGTTGCTAGCAGACAATTAGAA',
    'AGGTCTCTTTTGCTGATAGTCATTACATTCGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGGCTTCCTTCTTTACTAGGTCACTAGACGAAGCATCTGTTGACCTAGTTTTATGGGAGGAACTAACGGAATAAGTGCATTTGGCCTGAGTTCTTTCTTGTGAATAGGAATTGTGATTATTTCTAACTAATATTTTACTGGGGGTGGGGGAGCAGAGCTTTGTTTTAATGTAGC',
    'CCAAATGCCAGACAATGAAGTCATACAGCCTTGGCAGGTCTAAATATCACAGTTGTTCAAAAATGGGAGGAAGGGAGAGTGCCCCAGTTTTCAGAACTTCCGCTGAAAATGAACACAGAGAAGACCTACTTATCAGTTTCCTAACCAATGAGGTCACTGGGCAGGAAATCACCTTTTATAAAGAGTCAGTAAGCAGACTCTCCCTGCTTTGAATTTTGCATAAACAGCATTAGTGAGCACTTATTCCAGCC',
    'ACATGCCTCAGGTGATGGACTAAAGCTATAGCTGATGGCTAGCTCAGCTTTTACTCACTGTCAGCTGCATGCACAGATGTCTCCTTCAGTGGGGTTTAATACACAGTCCTTCATCCTCCAGCTATACATGGAACTGAAACCAGAGGCTCTGACTGCCATCTCCCATGTGGCTGCAGGATAGTTCTGTCTTTGGCCTTTACTGAGAGACAGACAGAGGTGTTAAGCCCTCTGCAGTGACTCTGGGAACCAGA']
    
    x = np.array([one_hot_encode(seq).numpy() for seq in seqs])
    
    Y_pred = model.forward(torch.Tensor(x))
    Y_pred = Y_pred.detach().cpu().numpy()
    ranks = []
    print('Highest activity of sequence in track')
    for i, yp in enumerate(Y_pred):
        print(i, np.argmax(yp), np.amax(yp))
    
    # Example sequence
    vis_seq = 5
    # Highest activity track for example sequence
    track = 22
    
    
    '''
    # A unifying concept for genomic sequence attribution scores
    
    ## Feature importance scores 
    
    Sequence-to-function models have become the preferred tool to analyze the relationship between genomic sequence and genome-wide experimental measurements such as chromatin accessibility 1,2, gene expression3–5, 3D chromatin conformation 6–8, and other molecular data modalities 9–11. These models usually consist of some sort of a convolutional neural network that operates on a one-hot encoded sequence to predict the measured phenotype. To better understand the models’ decision processes, and extract the learnt gene regulatory language, various feature attribution methods can be applied to these Deep Learning models 12–14. For genomic sequence-to-function models, these methods estimate the importance of each possible nucleotide within an input sequence to the model’s prediction. It is hoped that when applied to increasingly accurate sequence-based deep learning models, these methods can aid in solving for a comprehensive cis regulatory grammar and replace laborious and expensive in vivo experiments to identify regulatory sequences across the genome 4,9,15. 

    Feature attribution methods can be classified into perturbation-based (or sampling-based) algorithms and back-propagation-based algorithms 16. Both classes of methods approximate the non-linear neural network function f(s) around the sequence of interest s0 with a linear model within the regime of Δs. The linear model assigns a single multiplier ms0 to every input feature, from which one can attribute importances to the individual input features to the model’s predictions.
    f(s0+s)f(s0) + msos		(1)
    For example, to approximate the value of a complex function f at input s, the over 300 year old Taylor’s approximation linearly decomposes the function value f(s) at s into the value of the function at nearby position s0, given by  f(s0),  and the derivative of the function dfds0 at s0 multiplied by the difference between the position of interest s and s0.
    f(s0+s)f(s0) + dfds0s		(2)
    Taylor’s approximation represents a linear back propagation-based approximation in the infinitely small regime around the point s0 with the multiplier ms0 represented by the gradient dfds0 at s0. 
    To interpret these local linear approximations of the model, people derive either local or global importance scores from them 17. The local importance scores are simply given by the coefficients or multipliers of the linear approximation of the model. They determine how important an input feature is locally around the sequences. Global importance scores are given by the linear coefficients multiplied with the difference between the investigated sequence and a ‘neutral’ reference sequence. Global scores account for different scales of non-standardized features. 
    aglobal= msos	(3)
    alocal= mso		(4)
    
    '''
    # compute gradient (local attributions)
    t1 = time.time()
    grad_local = takegrad(x, model, tracks = track, output = 'local', device = None, baseline = None)
    t2 = time.time()
    print("Local attribution scores computed, aka gradients, in", '{}s'.format(round(t2-t1,3)), 'of shape', np.shape(grad_local))
    
    grad_local0 = grad_local[vis_seq,0]
    fig_local = plot_attribution(grad_local0, heatmap = grad_local0, ylabel = 'Grad\n(local)')
    
    # compute gradient times input ( global attributions to 0 reference)
    t1 = time.time()
    grad_global = takegrad(x, model, tracks = track, output = 'global', device = None, baseline = np.zeros(b))
    t2 = time.time()
    print("Global attribution scores computed, aka gradient times input, in", '{}s'.format(round(t2-t1,3)), 'of shape', np.shape(grad_global))
    
    grad_global0 = grad_global[vis_seq,0]
    fig_global = plot_attribution(grad_global0, heatmap = grad_global0, ylabel = 'Gradxinput\n(global)')
    
    '''
    Other backpropagation based methods approximate the linear function in a larger regime from sequence of interest to a baseline sequence sb. The baseline is often defined as a sequence with neutral prediction, or even more stringent as the “closest sequence” to the sequence with neutral prediction. These methods guarantee completeness, also known as efficiency. Completeness guarantees that the sum of all global attributions will be equal to the difference of the function between the baseline and the sequence of interest, a constraint of the feature attribution method that can be useful for some situations.
    aglobal= f=f(sb)-f(s0)		(5)
    The most comprehensible strategy to create an approximation is through sampling based methods that sample a sufficient number of data points between s and s0 and fit a generalized linear model to them. The choice of the sampling and generalized linear model defines the resulting multipliers and determines their interpretations. Sampling based methods can also be used to explicitly fit second, or even higher order effects from sufficient sampled data points18. However, sampling based methods suffer from the large computational cost that is required for generating sufficient data points. Perturbation methods represent a special case of sampling methods, which reduce the sampling cost to a minimum number of one sample per feature. Perturbation based methods disturb only a single feature at the time while leaving all others as they are to directly derive the varied features’ multiplier from the difference of models predictions to the original. 

    In computer vision, these perturbations are simply performed by setting pixels to zero. However, for genomic sequence based models which use one-hot encoded sequences of length L and width 4 (A,C,G,T bases), these perturbations are motivated by wet lab experiments, in which one base is exchanged for another. These so-called in silico mutagenesis (ISM) experiments, set the current base to zero but also add a one to another base at that position in the one-hot encoding15. It can be intuitively compared to performing  in vivo saturation mutagenesis experiments 16. The differences between the predictions of these variant sequences and the prediction from the “reference” (initial) sequence is then used to define the impact or importance of the reference base and each alternative base along the sequence. 

    ## Taylor approximation links gradients to ISM values
    Taylor directly links ISM to the linear approximation from the models’ gradients, or other backpropagation based methods. The gradient dfds0 can also be represented as the finite difference from the reference s0 to a sequence with a single nucleotide substitution from b0to b1 at position l, denoted by  s0(l,b0b1) .
    f(s)f(s0)+f(s0)ssf(s0)+f(s0(l,b0b1))-f(s0)s0(l,b0b1)-s0s		(6)
    In the case where the finite distance δs to approximate the gradient is equal to the distance from the reference sequence Δs, the numerator and denominator cancel each other out and we are left with the ISM value given by Equation (3).
    =f(s0)+f(s0(l,b0b1))-f(s0)s0(l,b0b1)-s0(s0(l,b0b1)-s0)		
    =f(s0)+f(s0(l,b0b1))-f(s0)=f(s0)+ISM(s0,l,b1)		(7)
    Thus, Equation (7) shows that ISM can be understood as the linear effect from an approximation of the deep learning model f  with respect to a surrogate variable s=s-s0. Vice versa, using this relationship enables us to approximate time consuming ISM values from the models’ gradient.
    f(s0)+ISM(s0,l,b1)f(s0)+dfds0s0(l,b0b1)-dfds0s0=f(s0)+TISM(s0,l,b1)   (8)
    where TISM denotes the first-order Taylor approximation to ISM. Applying this to a one-hot encoded input in which the reference base b0 a position l is replaced (set b0 from 1 to 0) by an alternative base b1 (set b1 from 0 to 1), we can see that the ISM at l,b1 is equal to the gradient with respect to the reference sequence s0 at base l,b1 minus the gradient at base l,b0. 
    ISM(s0,l,b1)df(l,b1)ds0-df(l,b0)ds0=TISM(s0,l,b1)	(9)
    Using this relationship allows us to quickly approximate the per nucleotide ISM values from the gradient of the input sequence using only a single forward pass through the model. 

    In practice, ISM values are used in two ways: (1) as a per-nucleotide value that indicates how much the prediction changes if the reference base is replaced by the specific variant; (2) as attribution maps which indicate how important each nucleotide is for a model’s prediction. To generate attribution maps from ISM values, practitioners use different methods. The most comprehensible method simply maps the mean effect of replacing a base with one of the three alternatives onto the reference base. This representation is then used to determine motifs, i.e. longer seqlets with subsequent bases with significant attributions.

    '''
    
    # compute taylor approximated in silico saturation mutagenesis effects.
    t1 = time.time()
    grad_tism = takegrad(x, model, tracks = track, output = 'tism', device = None, baseline = None)
    t2 = time.time()
    print("TISM values computed from gradients in", '{}s'.format(round(t2-t1,3)), 'of shape', np.shape(grad_tism))
    
    # TISM can be used just like ISM values, for example plotting the mean effect of mutating a base to determine the importance of that base.
    tism0 = grad_tism[vis_seq,0]
    meaneffect_tism = -np.sum(tism0/3, axis = -2)[None,:] * x[vis_seq]
    fig_tism = plot_attribution(meaneffect_tism, heatmap = tism0, ylabel = 'Mean\nTISM')


    
    # compute ISM effects
    t1 = time.time()
    ism_array = ism(x, model, tracks = track)
    t2 = time.time()
    print("ISM values computed in", '{}s'.format(round((t2-t1),3)), 'of shape', np.shape(ism_array))
    
    ism0 = ism_array[vis_seq,0]
    meaneffect_ism = -np.sum(ism0/3, axis = -2)[None,:] * x[vis_seq]
    fig_ism = plot_attribution(meaneffect_ism, heatmap = ism0, ylabel = 'Mean\nISM')
    
    # Compare ISM to TISM
    print('ISM versus TISM')
    for i in range(np.shape(x)[0]):
        print(i, pearsonr(grad_tism[i,0].flatten(), ism_array[i,0].flatten())[0]) #, pearsonr(mean_grad_tism[i,0], mean_grad_ism[i,0])
    
    '''
    #The hypothetical attribution score unifies different methods

    While local and global importance scores are common in image classification, a third attribution score has become more relevant for genomic sequence analysis, the “hypothetical attribution score”. “The hypothetical importance scores are meant to give a sense of what importance would be placed on a different base in the sequence if that base were present. We tend to think of it as an "autocomplete" of the scores, because in cases where you have a partial motif match, the hypothetical importance scores could give you a sense of what a stronger motif match would have looked like.” (https://github.com/kundajelab/tfmodisco/issues/5). This feature is beneficial to extract contribution weight matrices that, similar to position weight matrices, not only tell us about the contribution of the present feature but also about what would be preferred instead. The hypothetical contributions for a base are created by a transformation of the local attribution scores, which subtracts the sum over all bases of the difference between the baseline and the sequence of interest weighted by their local attributions from the attribution at the base, i.e. the difference between a sequence where the hypothetical base was present and the selected baseline sequence with base-pair probabilities b(i) at base i. 
    ahypo(j)= mso(j) -i{A,C,G,T}b(i)mso(i) 	; j{A,C,G,T} 	(10)
    Note that the hypothetical attribution at the reference is the same as the sum of the global attributions at the same loci. Hypothetical attributions determine the effect at the entire loci if we would go from the chosen baseline to a sequence with that base at this position, therefore hypothetical. This feature generally seems to better represent the learned sequence grammar of the model, and that’s why they have been introduced with TFmodisco, a widely used software package to extract and cluster sequence motifs from attribution maps. 

    For gradient-based attributions, we can quickly see that the recently suggested correction by Majdandzic et al. 19 of the gradient represents the hypothetical attribution score from a baseline with 0.25 uniform base-pair probability. 
    gc(j)=g(j)-14i= 14g(i)=mso(j) -i{A,C,G,T}0.25mso(i) =ahypouniform(j) 	(11)

    '''
    
    # compute corrected gradients, which can also be referred to as hypothetical attributions with uniform baseline
    t1 = time.time()
    grad_corrected = takegrad(x, model, tracks =track, output = 'corrected', device = None, baseline = None)
    t2 = time.time()
    print("Hypothetical attribution scores computed, aka corrected gradients, in", '{}s'.format(round(t2-t1,3)), 'of shape', np.shape(grad_corrected))
    
    grad_hypothetical_uniform = takegrad(x, model, tracks = track, output = 'hypothetical', device = None, baseline = np.ones(b)*0.25)
    
    # Hypothetical attribution scores to uniform baseline are equivalent to corrected gradients
    grad_corrected0 = grad_corrected[vis_seq,0]
    fig_corrected = plot_attribution(grad_corrected0, heatmap = grad_corrected0, ylabel = 'Corr_Grad\n(hypo)')
    grad_hypothetical_uniform0 = grad_hypothetical_uniform[vis_seq,0]
    fig_hypothetical = plot_attribution(grad_hypothetical_uniform0, heatmap = grad_hypothetical_uniform0, ylabel = 'Grad_uni\n(hypo)')
    
    '''
    To visualize and identify sequence motifs from ISM, ISM values at all loci are often centered around the mean effect, so that the effect of the reference base becomes the negative mean of the effects from replacing it with the other four bases. 
    IC(j) =I(j)-14i{A,C,G,T}I(i)		(12)
    Knowing that the ISM value I(j) of an alternative base j is the result of removing the reference base r and replacing it with the alternative base j, we can also write this procedure in the form of the coefficients of the linear approximation of the model. 
    I(j)= f(s0(rj))-f(so)mss		(13)
    Note that  represent the vector product. Since only the two above mentioned procedures are performed during ISM, this vector product simply reduces to:
    I(j)ms(j)-ms(r)	(14)
    With (10) inserted into (8), we can easily determine that indeed the centered ISM values are also representing the hypothetical attributions to a uniform baseline of 0.25. 
    IC(j) =I(j)-14i{A,C,G,T}I(i)=ms(j) -i{A,C,G,T}0.25mS(i) =ahypouniform(j) 	(15)

    '''
    # Hypothetical attributions to uniform baseline are equivalent to centered ISM 
    ism_hypothetical = correct_multipliers(ism_array, 'corrected')
    ism_hypo0 = ism_hypothetical[vis_seq,0]
    fig_ismhyp0 = plot_attribution(ism_hypo0, heatmap = ism_hypo0, ylabel = 'ISM_Centered\n(hypo)')
    
    '''
    With the definition of hypothetical attribution scores we can theoretically link all commonly used attribution methods for genomic sequence-to-function models. Previously practitioners used to process different methods intuitively. While previous work has provided unifying views on the different attribution generating methods (i.e. the algorithm that produces the multipliers), none of these has focused on the choice of reference or their downstream processing. Here, we explained how the default output of different methods can be different types of attributions, and how we can process the output of different methods consistently to make them comparable. Comparing different types of attribution scores does not make sense since distinct processing will make them different by definition. For example, comparison between gradients with centered ISM values cannot yield the same motifs since they represent two different things. 
    '''
    
    # Let's compare hypothetical attributions from gradients to hypothetical attributions from deeplift with captum
    t1 = time.time()
    grad_deeplift = deepliftshap(x, model, tracks = track, deepshap = False, baseline = torch.ones(4)*0.25)
    t2 = time.time()
    print("DeepLift local scores computed in", '{}s'.format(round(t2-t1,3)), 'of shape', np.shape(grad_deeplift))
    
    # Create hypothetical deeplift scores from local scores 
    grad_deeplift_hypothetical = correct_multipliers(grad_deeplift, 'hypotetical', x, np.ones_like(x)*0.25)
    grad_deeplift0 = grad_deeplift_hypothetical[vis_seq,0]
    fig_deeplift = plot_attribution(grad_deeplift0, heatmap = grad_deeplift0, ylabel = 'DeepLift\n(hypo)')
    
    
    # Now let's compare these attributions with each other

    # hypothetical scores from grad versus hypothetical scores from deeplift
    print('DeepLift versus Gradient hypothetical')
    for i in range(np.shape(x)[0]):
        print(i, pearsonr(grad_hypothetical_uniform[i,0].flatten(), grad_deeplift_hypothetical[i,0].flatten())[0])
    
    # Compare ISM to TISM
    print('DeepLift versus ISM')
    for i in range(np.shape(x)[0]):
        print(i, pearsonr(grad_deeplift_hypothetical[i,0].flatten(), ism_hypothetical[i,0].flatten())[0]) #, pearsonr(mean_grad_tism[i,0], mean_grad_ism[i,0])
    
    plt.show()
