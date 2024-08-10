import numpy as np
import sys, os

if __name__ == '__main__':
    file = sys.argv[1]
    outname = os.path.splitext(sys.argv[1])[0]
    data = np.load(file)
    print(data.files)
    x = data['values']
    names = data['names']
    print(np.shape(x), np.shape(names))
    if 'experiments' in data.files:
        exp = data['experiments']
        print(np.shape(exp))
        emeans = np.mean(x, axis = 0)
        estd = np.std(x, axis = 0)
        np.savetxt(outname + 'mexp.txt', np.array([exp, emeans, estd, estd/emeans]).T, fmt = '%s')

    nmeans = np.mean(x, axis = 1)
    nstd = np.std(x, axis = 1)

    np.savetxt(outname + 'mocr.txt', np.array([names, nmeans, nstd, nstd/nmeans]).T, fmt = '%s')


