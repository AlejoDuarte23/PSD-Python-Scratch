import pandas as pd
import seaborn as sns
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
# from scipy.signal import fftpack 
Acc1 = pd.read_csv('Vib2018-12-07(11_02_40)_1.csv')
Acc2 = pd.read_csv('Vib2018-12-07(110240)_2.csv')
Acc3 = pd.read_csv('Vib2018-12-07(11_02_40)_3.csv')
Acc4 = pd.read_csv('Vib2018-12-07(11_02_40)_4.csv')
Acc5 = pd.read_csv('Vib2018-12-07(11_02_41)_5.csv')





# sns.lineplot(x= Acc1['time (sec)'], y= Acc1['Z vibration (m/s^2)'])
# sns.lineplot(x= Acc2['time (sec)'], y= Acc2['Z vibration (m/s^2)'])
# sns.lineplot(x= Acc3['time (sec)'], y= Acc3['Z vibration (m/s^2)'])


# plt.plot(Acc1['time (sec)'], Acc1['Z vibration (m/s^2)'])
# for target in [Acc2, Acc3]:
#     dx = np.mean(np.diff(Acc1['time (sec)'].values))
#     shift = (np.argmax(signal.correlate(Acc1['Z vibration (m/s^2)'], target['Z vibration (m/s^2)'])) - len(target['Z vibration (m/s^2)'])) * dx
#     plt.plot(target['time (sec)'] + shift, target['Z vibration (m/s^2)'])
    
def transfor_g_ms(Data):
    Col_name = Data.columns
    for names in Col_name:
        if names != Col_name[0]:
            Data[names] = Data[names].values*9.80665
            Data = Data.rename(columns={names: f'{names[0]} vibration (m/s^2)'})
    # print(Data.columns)
    return Data


Acc2 = transfor_g_ms(Acc2)
Acc4 = transfor_g_ms(Acc4)


# plt.figure()
# def plot_dataset(dir):
#     # plt.figure()
#     for Data  in [Acc1,Acc2 ,Acc3, Acc4, Acc5]:
#         freq,Yo = signal.welch(Data[dir],95,nperseg=int(len(Data[dir])/4)+1)
#         plt.plot(freq,10*np.log10(Yo))
#     plt.legend(['SP1','SP2','SP3','SP4','SP5','SP1','SP2','SP3','SP4','SP5'])
    
    
# plot_dataset('Z vibration (m/s^2)')
# plot_dataset('Y vibration (m/s^2)')



H = {}

d = 0
fs = 95
for Data  in [Acc1,Acc2 ,Acc3, Acc4, Acc5]:
    c = 0
    h = np.array([])
    
    Col_name = Data.columns
    Fs = 1/np.mean(np.diff(Data[Col_name[0]]))
    N = Data[Col_name[0]].shape[0]
    n = int(N*fs/Fs)

    for names in Col_name:
        if names != Col_name[0]:
            _Acc = signal.resample(Data[names].values, n)
            if c == 0:
                h = _Acc
                # print(h.shape)
            else:
                h = np.column_stack((h,_Acc))

                # print(h.shape)
            c = c+1
    H[d] = h
    d = d+1
            
plt.close('all')
dx = 1/fs
SF = [0]
_min_ln= 0
for key,value in H.items():
    
    if key != 0:
        cor = signal.correlate(H[0][:,2],value[:,2])
        shift = (np.argmax(signal.correlate(H[0][:,2],value[:,2])) - len(value[:,2])) 
        print('samples_to_shift0',shift)
        SF.append(shift)
        if value[-shift:-1,2].shape < _min_ln:
            _min_ln = value[-shift:-1,2].shape
    else:
        _min_ln = value[:,2].shape
    print( _min_ln)
        
print('ssssss')
_ACC_95 = np.zeros((_min_ln[0],0))
print(_ACC_95.shape)
for key,value in H.items():
    print(value[abs(SF[key]):_min_ln[0]+abs(SF[key]),:].shape)
    _ACC_95 = np.column_stack((_ACC_95,value[abs(SF[key]):_min_ln[0]+abs(SF[key]),:]))
    
plt.plot(_ACC_95[5000:20001,:])




 


    
    
    # for target in [Acc2, Acc3]:
    #     dx = np.mean(np.diff(Acc1['time (sec)'].values))
    #     shift = (np.argmax(signal.correlate(Acc1['Z vibration (m/s^2)'], target['Z vibration (m/s^2)'])) - len(target['Z vibration (m/s^2)'])) * dx
    #     plt.plot(target['time (sec)'] + shift, target['Z vibration (m/s^2)'])
        
        

# A = fftpack.fft(a)
# B = fftpack.fft(b)
# Ar = -A.conjugate()
# Br = -B.conjugate()



# plot_dataset('X vibration (m/s^2)')
    
    
        