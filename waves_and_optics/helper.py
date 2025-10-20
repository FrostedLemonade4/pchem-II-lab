import ipywidgets as widgets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy

# Superposition Plot
def superposition_plot(f1, p1, f2, p2):
    fig, (ax1, ax2) = plt.subplots(2,1)
    x, y_1, y_2, y_superposition = waves_and_superposition(f1,p1,f2,p2)

    ax1.plot(x,y_1,label='First Wave',alpha=0.6,color='red')
    ax1.plot(x,y_2,label='Second Wave',alpha=0.6,color='orange')
    ax1.plot(x, y_superposition,label='Sum of waves',color='blue')
    ax1.set_title('Wave Superposition')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_ylim(-3, 3)
    ax1.set_xlim(0,4)

    fourier_x, fourier_y = fourier_transform(y_superposition)
    ax2.plot(fourier_x,fourier_y)
    ax2.set_title('Fourier Transform of Resulting Wave')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Contribution')
    ax2.set_xlim(0,6)
    ax2.set_ylim(0,1.05)

    plt.subplots_adjust(hspace=0.6)
    plt.show()

def waves_and_superposition(f1,p1,f2,p2):
    #creating list of x values for use with our functions
    x = np.linspace(0, 10, 640) 
    #transforming to angular frequencies for use in sine
    w1 = 2 * np.pi * f1
    w2 = 2 * np.pi * f2
   
    y_1 = np.sin(w1 * x + p1)
    y_2 = np.sin(w2 * x + p2)
    y_superposition = y_1 + y_2
    
    return x, y_1, y_2, y_superposition

def interactive_superposition_plot():
    return widgets.interact(superposition_plot, f1=(0,5,0.1), p1=(0,20,0.1),f2=(0,5,0.1),p2=(0,20,0.1))

def fourier_transform(signal):
    sampling_rate = 64  
    T = 1 / sampling_rate  
    
    n = len(signal)
    fft_signal = np.fft.fft(signal)
    fft_signal = fft_signal / n 
    
    frequencies = np.fft.fftfreq(n, T)
    
    positive_frequencies = frequencies[:n // 2]
    fft_signal_magnitude = np.abs(fft_signal[:n // 2])
    return positive_frequencies, fft_signal_magnitude