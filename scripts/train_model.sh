# Download mouse mm10 genome with
mkdir mm10
cd mm10
for i in {1..19}
do
wget --timestamping 'ftp://hgdownload.cse.ucsc.edu/goldenPath/mm10/chromosomes/chr'${i}'.fa.gz' -O chr${i}.fa.gz
done

# create fasta files with sequences of 250bp length
processdir=/home/to/DRG/scripts/train_models
python ${processdir}/generate_fasta_from_bedgtf_and_genome.py mm10/ ImmGenATAC1219.peak_matched.txt 

# Precompute one-hot encodings
python ${processdir}/transform_seqtofeature.py mm10ImmGenATAC1219.peak_matched.fasta



# Train model variants
ln -s mm10ImmGenATAC1219.peak_matched_onehot-ACGT_alignleft.npz seq250.npz
ln -s mouse_peak_heights.csv atac.csv
infile='seq250.npz' # one-hot encoded sequences of length 250bp, create softlink to rename to shorter version
outfile='atac.csv' # log-scaled atac-seq counts for each peak, create softlink to rename to shorter

del=',' # delimiter for outfile
cv=0 # crossvalidation set
bs=64 # batchsize

cuda=0 # gpu
i=1 # Seed for initialization

loss_function=Correlationmse # Training loss function 0.95*MSE+0.05*Correlation(peaks)
validation_loss=Correlationdata

num_kernels=298 # Number of convolutional kernels
l_kernels=19 # Length of convolutional kernels
kernel_function=GELU # Activation function of first layer
weighted_pooling=True # Weighted mean pooling layer, uses Softmax for weighted average
pooling_size=2 # Size of initial pooling layer
dilated_convolutions=4 # Number of additional residual convolutional blocks
l_dilkernels=7 # Size of convolutions in residual conv. blocks
dilweighted_pooling=2 # Weighted mean pooling size after each conv. block
fclayer_size=512 # Size of fully connected layers after conv. blocks
nfc_layers=2 # Number of additional fully connected layers
fc_dropout=0.1 # Drop out in fully connected layers
conv_batch_norm=True # batch norm in conv block

# baseline model architecture
baseline_model='loss_function=Correlationmse+validation_loss=Correlationdata+num_kernels=298+kernel_bias=False+l_kernels=19+kernel_function=GELU+max_pooling=False+weighted_pooling=True+pooling_size=2+dilated_convolutions=4+l_dilkernels=7+dilweighted_pooling=2+fclayer_size=512+nfc_layers=2+fc_dropout=0.1+conv_batch_norm=True'
# Training parameters
training='epochs=200+patience=15+lr=5e-7+batchsize='${bs}'+optimizer=SGD+optim_params=0.9+init_epochs=0+init_adjust=True+shift_sequence=10+random_shift=True+device=cuda:'${cuda}'+keepmodel=True+seed='${i}
# defined set of sequences that are in the validation and test set
cvset=../data/ImmGenATAC1219.peak_matchedtestsetschr81119.txt

traindir=/home/to/DRG/scripts/train_models

outdir=ATACmodels/

# Train model from with multiple initializations
for i in {1..5}
do
python ${traindir}/run_cnn_model.py ${infile} ${outfile} --delimiter $del --reverse_complement --crossvalidation $cvset $cv 10 --cnn ${baseline_model}+${training}+seed=${i} --addname 's'${i} --save_correlation_perclass --save_correlation_perpoint --outdir $outdir
done


i=1
# Model ablations
altmodel=dilpooling_residual=50 # Residual will be after 50 conv. blocks, since only 4 conv. blocks are initialized, there won't be any residuals
altmodel=loss_function=MSE # Use MSE only as loss function
altmodel=kernel_function=ReLU+net_function=ReLU # replace GELU with ReLU throughout the model
altmodel=kernel_function=EXP # Replace GELU after first convolutoinal layer with exponential function 
altmodel=max_pooling=True+weighted_pooling=False+dilweighted_pooling=0+dilmax_pooling=2 # Replace weighted mean pooling with max pooling
altmodel=reverse_complement=False # don't use reverse complement of sequence in first convolution
altmodel=fc_dropout=0. # don't use dropout
altmodel=conv_batch_norm=False # Don't use batch norm in conv. layers
altmodel=l1_kernel=0.01 # use a l1 penalty or kernels
altmodel=optimizer=AdamW+optim_params=None # use AdamW insted of SDG
altmodel=shift_sequence=None+random_shift=False # Don't augment data with shifted sequences.

# Train model ablation with
python ${traindir}/run_cnn_model.py ${infile} ${outfile} --delimiter $del --reverse_complement --crossvalidation $cvset $cv 10 --cnn ${baseline_model}+${training}+seed=${i}+${altmodel} --addname 's'${i} --save_correlation_perclass --save_correlation_perpoint --outdir $outdir

# Simple regression type cnn0
cnn0='loss_function=Correlationmse+validation_loss=Correlationdata+num_kernels=298+kernel_bias=False+l_kernels=19+kernel_function=GELU+max_pooling=False+weighted_pooling=True+pooling_size=70+nfc_layers=0+fc_dropout=0.1+conv_batch_norm=True'
python ${traindir}/run_cnn_model.py ${infile} ${outfile} --delimiter $del --reverse_complement --crossvalidation $cvset $cv 10 --cnn ${cnn0}+lr=1e-5 --addname 's'${i} --save_correlation_perclass --save_correlation_pergene --outdir $outdir 

# Train base model on subsets of the data
downsample='0.01 0.05 0.1 0.2 0.5'
for d in $downsample
do
echo $d
python ${traindir}/run_cnn_model.py ${infile} ${outfile} --delimiter $del --reverse_complement --crossvalidation $cvset $cv 10 --cnn ${baseline_model}+${training}+seed=${i} --addname 'ds'${d} --save_correlation_perclass --save_correlation_perpoint --outdir $outdir --downsample $d
done



