import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
plt.close('all')
#--------------- Code ------------------------#
def fft_np(Acc,fs):
    
    Nfft =  int(Acc.shape[0]/2)+1
    if Acc.shape[1] == 1:
        
        _Yxx = ((1/(fs*2*np.pi*Nfft))**0.5)*np.fft.fft(Acc)
        Yxx = _Yxx[:Nfft]
        frange = np.linspace(0, fs/2,Nfft)    
        return frange,Yxx
    else:
        _Yxx = ((1/(fs*2*np.pi*Nfft))**0.5)*np.fft.fft(Acc,axis = 0)
        Yxx = _Yxx[:Nfft,:]
        frange = np.linspace(0, fs/2,Nfft) 
        return frange,Yxx
    

def PSD_M(Fxx):
    NDT =  Fxx.shape[0]
    col =  Fxx.shape[1]
    PSD = np.zeros((NDT,col))
    TrPSD = np.zeros((NDT,1))
    for i in range(Fxx.shape[0]):
        Si = np.reshape(Fxx[i,:],(col,1))
        PSD[i] = np.diag(np.matmul(Si,np.conj(Si).T)).real
        TrPSD[i] = np.sum(PSD[i])/col
    return PSD,TrPSD

def Plot_PSD_M(frange,PSD,xo,xi,colors = ['c','m','b','k','y','r','g','c']):
    c= 0
    plt.figure()
    for psdi in np.hsplit(PSD,PSD.shape[1]):
        plt.plot(frange,psdi,color = colors[c])
        c = c+1
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('[Energy dB]')
    
    plt.yscale('log')
    plt.axis('tight')
    plt.xlim([xo,xi])

    plt.grid(True,which="both", linestyle='--')
    # plt.grid(True, which="both", ls="-", color='0.65')
    plt.show()

    
    
#--------------- Body ------------------------#
Acc1 = pd.read_csv('Vib2018-12-07(11_02_40)_3.csv')
fs = 100
_Acc = Acc1[['X vibration (m/s^2)','Y vibration (m/s^2)','Z vibration (m/s^2)']].to_numpy()
Acc = _Acc[2500:20000,:]
plt.figure()
plt.plot(Acc)
plt.figure()
frange,Yxx =  fft_np(Acc,fs)
PSD,TrPSD = PSD_M(Yxx)
Plot_PSD_M(frange,PSD,0.1,20)





#to do : Resample of the data 
#        Align data sets 
#       Get Modeshapes
#       Modeshape uncertainty 





           
           