import numpy as np
import matplotlib.pyplot as plt
import sys, os

def plotHist(x, bins = None, xlabel = None, title = None):
    fig = plt.figure(figsize = (3.5,3.5))
    axp = fig.add_subplot(111)
    a,b,c = axp.hist(x, color = 'navy', alpha = 0.5, bins = bins)
    axp2 = axp.twinx()
    axp2.spines['top'].set_visible(False)
    axp2.spines['left'].set_visible(False)
    axp2.tick_params(bottom = False, labelbottom = False)
    axp2.set_yticks([0.25,0.5,0.75,1])
    axp2.set_yticklabels([25,50,75,100])
    ag,bg,cg = axp2.hist(x, color = 'navy', alpha = 0.8, density = True, bins = bins, cumulative = -1, histtype = 'step')
    for p, per in enumerate(ag):
        print(round(bg[p],2), round(bg[p+1],2), round(per,2))
    
    axp.spines['top'].set_visible(False)
    axp.spines['right'].set_visible(False)
    if xlabel is not None:
        axp.set_xlabel(xlabel)
    return fig

if __name__ == '__main__':
    file = sys.argv[1]
    outname = os.path.splitext(sys.argv[1])[0]
    data = np.load(file)
    x = data['values'].flatten()
    print(np.mean(x), np.median(x))
    
    fig = plotHist(x, bins = np.linspace(0, 1, 101), xlabel='Correlation ISM vs Taylor')

    fig.savefig(outname + '_hist.jpg', dpi = 300, bbox_inches = 'tight')

