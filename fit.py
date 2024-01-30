import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def gauss(x,x0,sigma,A,y0):
    return A*np.exp(-pow(x-x0,2)/pow(sigma,2)) + y0

def fit(xdata,ydata,func=gauss):
    popt,_ = curve_fit(func, xdata, ydata,p0=[xdata[np.where(ydata==np.max(ydata))[0][0]],len(xdata)/8,np.max(ydata),np.min(ydata)])
    dataFit = gauss(xdata,*popt)
    return dataFit,popt

def fitFrames(lineouts):
    num_frames = lineouts.shape[0]
    fits = np.empty([num_frames,2],dtype='object')
    for i in np.linspace(0,num_frames-1,num_frames,dtype='int'):
         fits[i,0],fits[i,1] = fit(lineouts[i,0],lineouts[i,1])
    return fits

def plotFit(fit,lineout,figNum,save):
    plt.figure(figNum)
    plt.plot(lineout[0],lineout[1],label='Lineout')
    plt.plot(lineout[0],fit[0],label='Gaussian Fit')
    plt.grid()
    plt.legend()
    plt.xlabel('Pixels')
    plt.ylabel('Intensity')
    plt.title(f'Intensity versus Pixel for Frame {figNum}')
    if save:
        plt.savefig(f'lineout_fit_frame_{figNum}')

def plotFits(lineouts,fits,save=False):
    num_frames = lineouts.shape[0]
    for i in np.linspace(0,num_frames-1,num_frames,dtype='int'):
        plotFit(fits[i],lineouts[i],i,save)
    plt.show()