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

def findWidths(fits,mmppix,num_frames,units=0.001):
    widths = np.zeros([num_frames])
    for i in np.linspace(0,num_frames-1,num_frames,dtype='int'):
        widths[i] = units*mmppix*fits[i,1][1] #Save Gaussian width, rescale from mm to pixels 
    return widths

def computeVelocity(widths,frame_time,num_frames): #Since widths are in mm, arg "units" is a scaling constant letting the user set the spatial units of the velocity (i.e., for mm/s, use units = 1)
    vAvg = ((widths[-1]-widths[0])/(frame_time[-1]-frame_time[0]))
    vn = np.zeros([num_frames-1])
    print(f'\n\nWidths: {widths}\n\n')
    for i in np.linspace(1,num_frames-1,num_frames-1,dtype='int'):
        vn[i-1] = ((widths[i]-widths[i-1])/(frame_time[i]-frame_time[i-1]))
    return vn,vAvg

def plotVn(vn,vAvg,frame_time,units='m/s'):
    plt.plot(frame_time[1:],vn)
    plt.xlabel('Frame Times (s)')
    plt.ylabel(f'Velocities ({units})')
    plt.title('Average Velocity of Shock Between Time Steps')
    plt.text(frame_time[1],np.max(vn)*(9.9/10),f'Average Velocity: {round(vAvg,2)} {units}')
    plt.show()

def plotWidth(widths,frame_time,vAvg,units='m'):
    plt.plot(frame_time,widths)
    plt.xlabel('Frame Times (s)')
    plt.ylabel(f'Widths ({units})')
    plt.title(f'Spark Width as a Function of Time, Average Velocity = {round(vAvg,2)} {units}/s')
    plt.show()

def plotCalculations(widths,vn,vAvg,frame_time,units='m'):
    fig,ax = plt.subplots(1,2)
    fig.set_size_inches(18, 10, forward=True)

    ax[0].plot(frame_time[1:],vn)
    ax[0].set_xlabel('Frame Times (s)')
    ax[0].set_ylabel(f'Velocities ({units}/s)')
    ax[0].set_title('Average Velocity of Shock')
    ax[0].text(frame_time[1],np.max(vn)*(9.9/10),f'Average Velocity: {round(vAvg,2)} {units}/s')

    ax[1].plot(frame_time,widths)
    ax[1].set_xlabel('Frame Times (s)')
    ax[1].set_ylabel(f'Widths ({units})')
    ax[1].set_title(f'Spark Width, Average Velocity = {round(vAvg,2)} {units}/s')

    plt.show()