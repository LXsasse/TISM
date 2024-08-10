# Compute attributions with ISM and TISM

## Model ablations
model=ataconseq250_s1-cv10-0_Cormsek298l19FfGELUwei2vlCotasft101_dc4i1d1s1l7dw2r1nfc2s512tr1e-05SGD0.9bs64-F_model_params.dat # no reverse complement
model=ataconseq250rcomp_s1-cv10-0_Cormsek298l19FfEXPGELUwei2rcTvlCotasft101_dc4i1d1s1l7dw2r1nfc2s512cbnoTfdo0.1tr1e-05SGD0.9bs64-F_model_params.dat # EXP instead of GELU
model=ataconseq250rcomp_s1-cv10-0_Cormsek298l19FfGELUmax2rcTvlCotasft101_dc4i1d1s1l7da2r1nfc2s512tr1e-05SGD0.9bs64-F_model_params.dat # max pooling instead of weighted mean pooling
model=ataconseq250rcomp_s1-cv10-0_Cormsek298l19FfGELUwei2rcTvlCota_dc4i1d1s1l7dw2r1nfc2s512tr1e-05SGD0.9bs64-F_model_params.dat # no sequence shift
model=ataconseq250rcomp_s1-cv10-0_Cormsek298l19FfGELUwei2rcTvlCotasft101_dc4i1d1s1l7dw2r1nfc2s512fdo0.1tr1e-05SGD0.9bs64-F_model_params.dat # No batchnorm
model=ataconseq250rcomp_s1-cv10-0_Cormsek298l19FfGELUwei2rcTvlCotasft101_dc4i1d1s1l7dw2r1nfc2s512l1k0.0tr1e-05SGD0.9bs64-F_model_params.dat # No dropout
model=ataconseq250rcomp_s1-cv10-0_Cormsek298l19FfGELUwei2rcTvlCotasft101_dc4i1d1s1l7dw2r1nfc2s512tr1e-05AdamWbs64-F_model_params.dat # AdamW
model=ataconseq250rcomp_s1-cv10-0_Cormsek298l19FfGELUwei70rcTvlCotasft101tr1e-05SGD0.9bs64-F_model_params.dat # CNN0
model=ataconseq250rcomp_s1-cv10-0_Cormsek298l19FfReLUwei2rcTvlCotasft101_dc4i1d1s1l7dw2r1nfc2s512cbnoTfdo0.1tr1e-05SGD0.9bs64-F_model_params.dat # ReLU
model=ataconseq250rcomp_s1-cv10-0_MSEk298l19FfGELUwei2rcTvlCotasft101_dc4i1d1s1l7dw2r1nfc2s512cbnoTfdo0.1tr1e-05SGD0.9bs64-F_model_params.dat # MSE
model=ataconseq250rcomp_s1-cv10-0_Cormsek298l19FfGELUwei2rcTvlCotasft101_dc4i1d1s1l7dw2r1nfc2s512cbnoTfdo0.1tr1e-05SGD0.9bs64-F_model_params.dat # baseline model
model=ataconseq250rcomp_s1-cv10-0_Cormsek298l19FfGELUwei2rcTvlCotasft101_dc4i1d1s1l7dw2r1nfc2s512tr1e-05SGD0.9bs64-F_model_params.dat # baseline seed 1
model=ataconseq250rcomp_s2-cv10-0_Cormsek298l19FfGELUwei2rcTvlCotasft101_dc4i1d1s1l7dw2r1nfc2s512tr1e-05SGD0.9bs64-F_model_params.dat # baseline seed 2
model=ataconseq250rcomp_s3-cv10-0_Cormsek298l19FfGELUwei2rcTvlCotasft101_dc4i1d1s1l7dw2r1nfc2s512tr1e-05SGD0.9bs64-F_model_params.dat # baseline seed 3
model=ataconseq250rcomp_s4-cv10-0_Cormsek298l19FfGELUwei2rcTvlCotasft101_dc4i1d1s1l7dw2r1nfc2s512tr1e-05SGD0.9bs64-F_model_params.dat # baseline seed 4
model=ataconseq250rcomp_s5-cv10-0_Cormsek298l19FfGELUwei2rcTvlCotasft101_dc4i1d1s1l7dw2r1nfc2s512tr1e-05SGD0.9bs64-F_model_params.dat # baseline seed 5

# Models from down sampling
model=ataconseq250rcomp_ds0.01-cv10-0_Cormsek298l19FfGELUwei2rcTvlCotasft101_dc4i1d1s1l7dw2r1nfc2s512cbnoTfdo0.1tr1e-05SGD0.9bs64-F_model_params.dat
model=ataconseq250rcomp_ds0.05-cv10-0_Cormsek298l19FfGELUwei2rcTvlCotasft101_dc4i1d1s1l7dw2r1nfc2s512cbnoTfdo0.1tr1e-05SGD0.9bs64-F_model_params.dat
model=ataconseq250rcomp_ds0.1-cv10-0_Cormsek298l19FfGELUwei2rcTvlCotasft101_dc4i1d1s1l7dw2r1nfc2s512cbnoTfdo0.1tr1e-05SGD0.9bs64-F_model_params.dat
model=ataconseq250rcomp_ds0.2-cv10-0_Cormsek298l19FfGELUwei2rcTvlCotasft101_dc4i1d1s1l7dw2r1nfc2s512cbnoTfdo0.1tr1e-05SGD0.9bs64-F_model_params.dat
model=ataconseq250rcomp_ds0.5-cv10-0_Cormsek298l19FfGELUwei2rcTvlCotasft101_dc4i1d1s1l7dw2r1nfc2s512cbnoTfdo0.1tr1e-05SGD0.9bs64-F_model_params.dat

# Run with each model
infile='seq250.npz' # one-hot encoded sequences of length 250bp
outfile='atac.csv' # log-scaled atac-seq counts for each peak
cvset=ImmGenATAC1219.peak_matchedtestsetschr81119.txt
traindir=/home/to/DRG/scripts/train_models
del=',' # delimiter for outfile
cv=0 # crossvalidation set
bs=64 # batchsize
outdir=ATACmodels/

python ${traindir}/run_cnn_model.py ${infile} ${outfile} --delimiter $del --reverse_complement --crossvalidation $cvset $cv 10 --cnn ${model} --sequence_attributions ism all zero_mean_gauge=False --outname ${outdir}Load 
python ${traindir}/run_cnn_model.py ${infile} ${outfile} --delimiter $del --reverse_complement --crossvalidation $cvset $cv 10 --cnn ${model} --sequence_attributions grad all zero_mean_gauge=False --outname ${outdir}Load

# For visualization, select individual sequences and only compute for these and the selected cell types
select_file=selection.txt # Text file that contains all names of peaks that should be used
zero_mean_gauge=False # Returns gradients for 'grad' and ISM values for ism. Grad values need to be transformed to TISM values before being compared to ISM

python ${traindir}/run_cnn_model.py ${infile} ${outfile} --delimiter $del --reverse_complement --select_list $select_file --predictnew --crossvalidation $cvset $cv 10 --cnn ${model} --sequence_attributions ism all zero_mean_gauge=False --outname ${outdir}Loadlist
select_peak=ImmGenATAC1219.peak_251276
python ${traindir}/run_cnn_model.py ${infile} ${outfile} --delimiter $del --reverse_complement --select_sample $select_peak --predictnew --crossvalidation $cvset $cv 10 --cnn ${model} --sequence_attributions grad all zero_mean_gauge=False --outname ${outdir}Load${select_peak}


