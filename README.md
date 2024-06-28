# TISM

## Summary
To understand the decision process of genomic sequence-to-function models, explainable AI algorithms determine the importance of each nucleotide in a given input sequence to the model’s predictions, and enable discovery of cis-regulatory motifs for gene regulation. The most commonly applied method is in silico saturation mutagenesis (ISM) because its per-nucleotide importance scores can be intuitively understood as the computational counterpart to in vivo saturation mutagenesis experiments. While ISM is highly interpretable, it is computationally challenging to perform for a large number of sequences, and becomes prohibitive as the length of the input sequences and size of the model grows. Here, we use the first-order Taylor approximation to approximate ISM values from the model’s gradient, which reduces its computation cost to a single forward pass for an input sequence. We show that the Taylor ISM (TISM) approximation is robust across different model ablations, random initializations, training parameters, and data set sizes. 

See our manuscript for more info [[1]](#1)

## Installation

Download the repository and setup conda environment.

Install by navigating to the location of the local repository

`pip install -e .`

## Usage

In this example, we're using the precomputed DeepSEA model, which can be downloaded from https://github.com/kipoi/models/tree/master/DeepSEA/beluga, simply with:

```
mkdir data
cd data
wget https://zenodo.org/record/3402406/files/deepsea.beluga.pth
```

Taylor's approximation represents a linear approximation in the infinitely small regime around the sequence of interest $`s_0`$ with the multiplier or coefficient of the linear model represented by the gradient $`df/ds_0`$. To interpret these local linear approximations of the model, people derive either ***local*** or ***global*** importance scores from them. The local importance scores are simply given by the coefficients or multipliers of the linear approximation of the model. They determine how important an input feature is locally around the sequences. Global importance scores are given by the linear coefficients multiplied with the difference between the investigated sequence and a "neutral" reference sequence. Global scores account for different scales of non-standardized features. 
    
```math
a_{local} = m_{s_0}
```
```math
a_{global} = m_{s_0} \cdot (s_0 - s_{baseline})
```

```python
import numpy as np
import torch
import matplotlib.pyplot as plt

from tism.models import Beluga
from tism.utils import plot_attribution, ism, deepliftshap
from tism.torch_grad import correct_multipliers, takegrad
from tangermeme.utils import random_one_hot

parameters = '../data/deepsea.beluga.pth'
model = Beluga()
model.load_state_dict(torch.load(parameters))

N=1
b=4
input_length = 2000
    
x = random_one_hot((N, b, input_length), random_state = 1).type(torch.float32)
x = substitute(x, "CTCAGTGATG")
x = x.detach().cpu().numpy()
    
track = 267
vis_seq = 0

grad_local = takegrad(x, model, tracks = track, output = 'local', device = None, baseline = None)
grad_local0 = grad_local[vis_seq,0][...,900:1100]
fig_local = plot_attribution(grad_local0, heatmap = grad_local0, ylabel = 'Grad\n(local)')
```
![image](https://github.com/LXsasse/TISM/blob/main/results/Local_attributions_gradient.jpg)

```python
grad_global = takegrad(x, model, tracks = track, output = 'global', device = None, baseline = np.zeros(b))
grad_global0 = grad_global[vis_seq,0][...,900:1100]
fig_global = plot_attribution(grad_global0, heatmap = grad_global0, ylabel = 'Gradxinput\n(global)')
```
![image](https://github.com/LXsasse/TISM/blob/main/results/Global_attributions_gradient.jpg)

### Taylor approximation links gradients to ISM values

Taylor directly links ISM to the linear approximation from the models' gradients, or other backpropagation based methods. The gradient $`df/ds_0`$ can also be represented as the finite difference from the reference $`s_0`$ to a sequence with a single nucleotide substitution from $`b_0`$ to $`b_1`$ at position l, denoted by  $`s_0(l,b_0,b_1)`$.
ISM can be understood as the linear effect from an approximation of the deep learning model f. Vice versa, using this relationship enables us to approximate time consuming ISM values from the models' gradients. Applying this relationship to a one-hot encoded input in which the reference base $`b_0`$ a position l is replaced (set $`b_0`$ from 1 to 0) by an alternative base $`b_1`$ (set $`b_1`$ from 0 to 1), we can see that the ISM at l, $`b_1`$ is equal to the gradient with respect to the reference sequence $`s_0`$ at base  l, $`b_1`$ minus the gradient at base  l, $`b_0`$. 

```math    
ISM(s_0,l,b_1) \approx \frac{df(l,b_1)}{ds_0}-\frac{df(l,b_0)}{ds_0}=TISM(s_0,l,b_1)
```    

Using this relationship allows us to quickly approximate the per nucleotide ISM values from the gradient of the input sequence with a single forward pass through the model. 

In practice, ISM values are used in two ways: (1) as a per-nucleotide value that indicates how much the prediction changes if the reference base is replaced by the specific variant; (2) as attribution maps which indicate how important each nucleotide is for a model’s prediction. To generate attribution maps from ISM values, practitioners use different methods. The most comprehensible method simply maps the mean effect of replacing a base with one of the three alternatives onto the reference base. This representation is then used to determine motifs, i.e. longer seqlets with subsequent bases with significant attributions.

```python
grad_tism = takegrad(x, model, tracks = track, output = 'tism', device = None, baseline = None)
tism0 = grad_tism[vis_seq,0]
meaneffect_tism = -np.sum(tism0/3, axis = -2)[None,:] * x[vis_seq]
fig_tism = plot_attribution(meaneffect_tism[...,900:1100], heatmap = tism0[...,900:1100], ylabel = 'Mean\nTISM')
```
![image](https://github.com/LXsasse/TISM/blob/main/results/TISM_mean.jpg)

```python
ism_array = ism(x, model, tracks = track, start = 900, end = 1100)
ism0 = ism_array[vis_seq,0]
meaneffect_ism = -np.sum(ism0/3, axis = -2)[None,:] * x[vis_seq]
fig_ism = plot_attribution(meaneffect_ism[...,900:1100], heatmap = ism0[...,900:1100], ylabel = 'Mean\nISM')
```
![image](https://github.com/LXsasse/TISM/blob/main/results/ISM_mean.jpg)

### Hypothetical attribution scores
Simply put, TISM links the model's gradient with values from ISM. These values can then be transformed in the user's preferred way and used for model interpretation. 
While local and global importance scores are common in image classification, a third attribution score has become more relevant for genomic sequence analysis, the “hypothetical attribution score”. 
> “The hypothetical importance scores are meant to give a sense of what importance would be placed on a different base in the sequence if that base were present. We tend to think of it as an "autocomplete" of the scores, because in cases where you have a partial motif match, the hypothetical importance scores could give you a sense of what a stronger motif match would have looked like.” (https://github.com/kundajelab/tfmodisco/issues/5). 

This feature is beneficial to extract contribution weight matrices that, similar to position weight matrices, not only tell us about the contribution of the present feature but also about what would be preferred instead. The hypothetical contributions for a base are created by a transformation of the local attribution scores, which subtracts the sum over all bases of the difference between the baseline and the sequence of interest weighted by their local attributions from the attribution at the base, i.e. the difference between a sequence where the hypothetical base was present and the selected baseline sequence with base-pair probabilities $`b(i)`$ at base i. 

```math
a_{hypo}(j)= m_{s_0}(j) - \sum_{i}^{\{A,C,G,T\}} b(i) \cdot m_{s_0}(i) \; ; \: j \in \{A,C,G,T\}
```
For gradient-based attributions, we can determine that the recently suggested correction by Majdandzic et al. [[2]](#2) of the gradient represents the hypothetical attribution score to a baseline with 0.25 uniform base-pair probability. 

```python
# compute corrected gradients, which can also be referred to as hypothetical attributions with uniform baseline
grad_corrected = takegrad(x, model, tracks =track, output = 'corrected', device = None, baseline = None)
grad_hypothetical_uniform = takegrad(x, model, tracks = track, output = 'hypothetical', device = None, baseline = np.ones(b)*0.25)
grad_corrected0 = grad_corrected[vis_seq,0]
fig_corrected = plot_attribution(grad_corrected0[...,900:1100], heatmap = grad_corrected0[...,900:1100], ylabel = 'Corr_Grad\n(hypo)')
```
 
![image](https://github.com/LXsasse/TISM/blob/main/results/Corrected_gradients.jpg)
   

To visualize and identify sequence motifs from ISM, ISM values at all loci are often centered around the mean effect, so that the effect of the reference base becomes the negative mean of the effects from replacing it with the other four bases.If we use the linear approximation from Taylor, We can easily determine that the centered ISM values are also representing the hypothetical attributions to a uniform baseline of 0.25. 

```python
# Hypothetical attributions to uniform baseline are equivalent to centered ISM 
ism_hypothetical = correct_multipliers(ism_array, 'corrected')
ism_hypo0 = ism_hypothetical[vis_seq,0]
fig_ismhyp0 = plot_attribution(ism_hypo0[...,900:1100], heatmap = ism_hypo0[...,900:1100], ylabel = 'ISM_Centered\n(hypo)')
```
![image](https://github.com/LXsasse/TISM/blob/main/results/CenteredISM_hypothetical.jpg)

### Time

While TISM and ISM correlated around 0.7, the speed up from TISM to ISM is massive. In practice, the speed up seems to be close to the theorical value of three times the length of the input sequence.

![image](https://github.com/LXsasse/TISM/blob/main/results/Comparison_time_hor_N_cpu.jpg)
![image](https://github.com/LXsasse/TISM/blob/main/results/Comparison_time_hor_L_cpu.jpg)

<!-- <img src="https://github.com/LXsasse/TISM/blob/main/results/Comparison_time_N_cpu.jpg" width="500"> this is a comment -->

## References
<a id="1">[1]</a> 
Quick and effective approximation of in silico saturation mutagenesis experiments with first-order Taylor expansion
Alexander Sasse, Maria Chikina, Sara Mostafavi,bioRxiv 2023.11.10.566588; doi: https://doi.org/10.1101/2023.11.10.566588 

<a id="2">[2]</a>
Majdandzic, A., Rajesh, C. & Koo, P.K. Correcting gradient-based interpretations of deep neural networks for genomics. Genome Biol 24, 109 (2023). https://doi.org/10.1186/s13059-023-02956-3
