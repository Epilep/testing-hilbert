import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftshift
from scipy.signal import hilbert

def man_hilbert(x):
    Xn = np.fft.fft(x) # fft via numpy
    Xs = fft(x) # fft via scipy
    N = len(Xn)
    Zn = np.zeros(N, dtype=np.complex)
    Zn[1:N//2] = Xn[1:N//2] * 1 # essa e a próxima implementam
    Zn[N//2:] = Xn[N//2:] * -1 # sign(w)
    Zs = np.zeros(N, dtype=np.complex)
    Zs[1:N//2] = Xs[1:N//2] * 1 
    Zs[N//2:] = Xs[N//2:] * -1 
    Hn = np.fft.ifft(Zn)
    Hs = ifft(Zs)
    In = np.fft.ifft(Xn) # FFT Inversa de X
    Is = ifft(Xs)
    return Hn, Hs, Xn, Xs, In, Is, Zn, Zs

freq = 0.0001
t = np.arange(start=0, stop=1, step=freq) # array com os instantes de tempo


x = np.cos(2*np.pi*10*t) # array com o sinal calculado no tempo t

Hn, Hs, Xn, Xs, In, Is , Zn, Zs = man_hilbert(x)

H = hilbert(x)

Ht = np.sin(2*np.pi*10*t) # transformada de hilbert teórica de cos

N  = len(Xn)

z = range(-N//2,N//2)

plt.subplot(311)
plt.plot(t,x, color='black', lw = 2)
plt.plot(t,In, color='blue', ls = '--', lw = 2)
plt.plot(t,Is, color='red', ls = ':', lw = 2)

plt.subplot(312)
plt.plot(t,Ht, color='black', lw = 2)
plt.plot(t,np.imag(Hn), color='blue', ls = '--', lw = 2)
plt.plot(t,np.imag(Hs), color='red', ls = ':', lw = 2)
plt.plot(t,np.imag(H), color='green', ls = '-.', lw = 2)

plt.subplot(313)
#plt.plot(x, color='black', label='cos(2*pi*10*t)', lw = 2)
plt.plot(z, np.fft.fftshift(Xn), color='blue', ls = '-', lw = 2)
plt.plot(z, fftshift(Xs), color='red', ls = ':', lw = 2)
plt.plot(z, np.fft.fftshift(Zn), color='black', ls = '--', lw = 2)
plt.plot(z, fftshift(Zs), color='green', ls = '-.', lw = 2)
plt.xlim(-15,15)
plt.show()


x = np.cos(2*np.pi*10*t) + 0.5*np.cos(2*np.pi*3*t) # array com o sinal calculado no tempo t

Hn, Hs, Xn, Xs, In, Is , Zn, Zs = man_hilbert(x)

H = hilbert(x)

Ht =  np.sin(2*np.pi*10*t) + 0.5*np.sin(2*np.pi*3*t) # transformada de hilbert teórica de cos

N  = len(Xn)

z = range(-N//2,N//2)

plt.subplot(311)
plt.plot(t,x, color='black', lw = 2)
plt.plot(t,In, color='blue', ls = '--', lw = 2)
plt.plot(t,Is, color='red', ls = ':', lw = 2)

plt.subplot(312)
plt.plot(t,Ht, color='black', lw = 2)
plt.plot(t,np.imag(Hn), color='blue', ls = '--', lw = 2)
plt.plot(t,np.imag(Hs), color='red', ls = ':', lw = 2)
plt.plot(t,np.imag(H), color='green', ls = '-.', lw = 2)

plt.subplot(313)
#plt.plot(x, color='black', label='cos(2*pi*10*t)', lw = 2)
plt.plot(z, np.fft.fftshift(Xn), color='blue', ls = '-', lw = 2)
plt.plot(z, fftshift(Xs), color='red', ls = ':', lw = 2)
plt.plot(z, np.fft.fftshift(Zn), color='black', ls = '--', lw = 2)
plt.plot(z, fftshift(Zs), color='green', ls = '-.', lw = 2)
plt.xlim(-15,15)
plt.show()

# x = np.ones(len(t), dtype=np.complex)
# for i in range(len(t)):
#     x[i] = np.cos(np.exp(1j * t[i])) # array com o sinal calculado no tempo t

x = np.real(np.exp(1j*10*t))
ft = (10*t+np.pi/2) % np.pi - np.pi/2
Hn, Hs, Xn, Xs, In, Is , Zn, Zs = man_hilbert(x)

H = hilbert(x)

#Ht = - np.sin(2*np.pi*10*t) - 0.5*np.sin(2*np.pi*3*t) # transformada de hilbert teórica de cos

fn = np.arctan(np.imag(Hn)/x) #achei que precisava de -
fs = np.arctan(np.imag(Hs)/x)
f = np.arctan(np.imag(H)/x)


plt.plot(t,ft, color='black', lw = 2)
plt.plot(t,fn, color='blue', ls = '--', lw = 2)
plt.plot(t,fs, color='red', ls = ':', lw = 2)
plt.plot(t,f, color='green', ls = '-.', lw = 2)

plt.show()

x = np.real(np.exp(1j*(10*t)**2))
ft = ((10*t)**2 + np.pi/2) % np.pi - np.pi/2
Hn, Hs, Xn, Xs, In, Is , Zn, Zs = man_hilbert(x)

H = hilbert(x)

#Ht = - np.sin(2*np.pi*10*t) - 0.5*np.sin(2*np.pi*3*t) # transformada de hilbert teórica de cos

fn = np.arctan(np.imag(Hn)/x) #achei que precisava de -
fs = np.arctan(np.imag(Hs)/x)
f = np.arctan(np.imag(H)/x)


plt.plot(t,ft, color='black', lw = 2)
plt.plot(t,fn, color='blue', ls = '--', lw = 2)
plt.plot(t,fs, color='red', ls = ':', lw = 2)
plt.plot(t,f, color='green', ls = '-.', lw = 2)

plt.show()

x1 = np.cos(2*np.pi*10*t) 
x2 = np.cos(2*np.pi*20*t) 
x = np.array([x1,x2])
print(x)
print(x.shape)
N  = len(x1)
z = range(-N//2,N//2)

f = fft(x)#,axis=0)
print(f)
print(f.shape)
plt.plot(z,fftshift(f[0]), color='black', lw = 2)
plt.plot(z,fftshift(f[1]), color='blue', ls = '--', lw = 2)
# plt.plot(t,fs, color='red', ls = ':', lw = 2)
# plt.plot(t,f, color='green', ls = '-.', lw = 2)
plt.xlim(-22,22)
plt.show()


x1 = np.real(np.exp(1j*10*t))
x2 = np.real(np.exp(1j*20*t))
x = np.array([x1,x2])
H = hilbert(x)
f = np.arctan(np.imag(H)/x)
plt.plot(t,f[0], color='black', lw = 2)
plt.plot(t,f[1], color='blue', ls = '--', lw = 2)
plt.show()
