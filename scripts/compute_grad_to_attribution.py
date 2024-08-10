import numpy as np
import sys, os

if __name__ == '__main__':
    ism = np.load(sys.argv[1], allow_pickle = True) # gradient values
    ref = np.load(sys.argv[2], allow_pickle = True) # onehot encoded sequences

    names, values, exp = ism['names'], ism['values'], ism['experiments']
    seqfeatures, genenames = ref['seqfeatures'], ref['genenames']
    
    # old version of DRG saved onehot encoded sequences differently
    if len(np.shape(seqfeatures)) == 1:
        seqfeatures, featurenames = seqfeatures
    # sort against each other
    nsort = np.argsort(names)[np.isin(np.sort(names), genenames)]
    gsort = np.argsort(genenames)[np.isin(np.sort(genenames), names)]
    values = values[nsort]
    seqfeatures = seqfeatures[gsort]
    names, genenames = names[nsort], genenames[gsort]
    
    values = np.transpose(values, axes = (0,1,3,2))
    print(np.shape(values), np.shape(seqfeatures)) # check if shapes match

    ref0 = np.zeros(np.shape(values))
    for i in range(np.shape(ref0)[0]):
        # find the reference bases
        where = np.where(seqfeatures[i] == 1)
        for j in range(np.shape(ref0)[1]):
            ref0[i,j,where[0]] = values[i, j, where[0], where[1]].reshape(-1,1)
        
    taylor = values - ref0 # adjust gardients
    
    np.savez_compressed(os.path.splitext(sys.argv[1])[0].rsplit('_grad',1)[0] + '_taylor'+sys.argv[1].rsplit('_grad')[1], values = np.swapaxes(taylor,-1,-2), names = names, experiments = exp)



