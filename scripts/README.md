This directory contains scripts to reproduce the results and figures shown in the paper. 

To build and run these models, go to 
> https://github.com/LXsasse/DRG/

Clone the repository and install with: 
```
pip install -e .
```

The processed data files can be downloaded using the following links:
  - [bed file containing peak locations](https://www.dropbox.com/s/r8drj2wxc07bt4j/ImmGenATAC1219.peak_matched.txt?dl=0)
  - [csv file containing measured ATAC-seq peak heights](https://www.dropbox.com/s/7mmd4v760eux755/mouse_peak_heights.csv?dl=0)

The data was generated in [Yoshida et al. 2019](https://doi.org/10.1016/j.cell.2018.12.036)

Follow steps described in to reproduce the results:

```
train_model.sh
```
```
compute_attributions.sh
```
```
analyze.sh
```
```
tism_time.py
```