import  streamlit_vertical_slider  as svs
from Signal_Generation_class import Signal
import numpy as np
import pandas as pd
import librosa
import librosa.display
from numpy.fft import fft,rfft,rfftfreq,irfft,fftfreq
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
#_______________Global Variables/functions for generation of synthetic signal(Sum of pure frequencies)__________________#
signal_default_time = np.arange(0,1,0.001)    #1000 default samples for the time axis   

signal_default_values = np.zeros(len(signal_default_time))  

Final_signal_sum = None       

total_signals_list = [Signal(amplitude=1,frequency=1,phase=0)]  #contains all individual signals (freq components) forming the final or resulted signal 

def Generate_syntheticsignal():
    global Final_signal_sum
    
    for signal in total_signals_list:
        Final_signal_sum += signal.amplitude * np.cos(2*np.pi*signal.frequency*signal_default_time + signal.phase*np.pi )
       
    Final_signal_data={'Time':signal_default_time, 'Amplitude':Final_signal_sum}
    Final_sig_dataframe = pd.DataFrame(Final_signal_data)
    return Final_sig_dataframe



def addSignalToList(amplitude, frequency, phase):
    """
    Add signals to added_list
    :param amplitude: the amplitude of the signal
    :param frequency: the frequency of the signal
    :param phase: the phase of the signal
    """
   
    signal = Signal(amplitude=amplitude, frequency=frequency, phase=phase)
    total_signals_list.append(signal)
   
def removeSignalFromList(amplitude, frequency, phase):
    
    """
    remove signals from added_list
    Parameters
    ----------
    amplitude : float
    the amplitude of the signal
    frequency : float
    the frequancy of the signal
    phase : float
    the phase of the signal
    """

    for signals in total_signals_list:
        if signals.amplitude==amplitude and signals.frequency == frequency and signals.phase == phase:
            total_signals_list.remove(signals)     


def get_Total_signal_list():
    return total_signals_list


def SignalListClean():
   
   total_signals_list.clear()
   
#________________________End of functions/ variables for synthetic signal generation__________________________________________________________________________________# 



def generate_vertical_sliders(array_slider_labels, array_slider_values,Slider_step=1):
    """
    Generate vertical sliders for different equalizer modes
    Parameters
    ----------
    Slider_labels : label for each slider that controls the magnitude of certain frequency 
    ranges for the different modes 
    
    Slider_values: factor that would be multiplied with the magnitude of some frequency 
    components
    
    Slider_step: step of increment/decrement for each slider
   
    Return : different_slider_values : has (Start,end,step) for all sliders in the selected mode
    
    """
    different_slider_values = []
    slider_columns = st.columns(len(array_slider_labels))
    for col_number in range(len(array_slider_labels)) :
        with slider_columns[col_number] :
            current_slider = array_slider_values[col_number]
            slider = svs.vertical_slider(key=array_slider_labels[col_number],min_value=current_slider[0],max_value=current_slider[1],
                                         default_value=current_slider[2],step=Slider_step)
            different_slider_values.append(slider)
            st.write(array_slider_labels[col_number]) 
            
    return different_slider_values         


def load_audio_file(path_file_upload):
    
    """
    Function to upload audio file given file path using librosa
    
    (Librosa is a Python package for analyzing and working with audio files,
    and it can handle a variety of audio file formats, including WAV, MP3, FLAC, OGG, 
    and many more.)
    
    Parameters:
    Audio file path
    
    Output:
    Audio samples
    Sampling rate
    """
    if path_file_upload is not None:
        audio_samples,sampling_rate=librosa.load(path_file_upload)
        
    return audio_samples,sampling_rate    

def Fourier_Transform_explicit(amplitude_signal,sampling_rate):
    """
    Nyquist rate = 2fmax
    Nyquist_freq = 1/2*sampling_freq
    
    FFT--->Output complex array that corresponds to the fourier transformed signal
    It symmetric about the nyquist frequency , so we take the magnitude of the positive 
    part and multiply it by 2 except for the DC component
    and the nyquist frequency (To accomodate for the negative part)
    """
    number_of_samples = len(amplitude_signal)
    
    Sampling_period = 1/sampling_rate
    
    
    fft_result = fft(amplitude_signal,number_of_samples) # Compute the one-dimensional DFT using FFT Algorithm.
    
    magnitude_frequency_components = np.abs(fft_result) #Gets the magnitude of frequency components
    
    positive_frequencies = magnitude_frequency_components[:number_of_samples//2 +1]  #  Take positive part only
     
    positive_frequencies[1:-1] *=2  #Multiply magnitude by two except for DC component and Nyquist frequency
      
    frequency_components = fftfreq (number_of_samples,Sampling_period)[:number_of_samples//2 +1]   #Get corresponding frequencies of the magnitude of the FFT
    
    return magnitude_frequency_components, frequency_components


def Fourier_Transform_Signal(amplitude_signal, sampling_rate):
    """
    rrft-->  specialized version of the FFT algorithm that is 
    optimized for real-valued signals,  returns only positive frequencies
    
    rfftfreq--> function from the NumPy library is used to compute the frequencies 
    directly from the signal, returns an array of frequencies corresponding to the 
    output of the rfft function. 
    """
    number_of_samples = len(amplitude_signal)
    
    sampling_period = 1/sampling_rate
    
    magnitude_freq_components = rfft(amplitude_signal)
    
    frequency_components = rfftfreq(number_of_samples,sampling_period)
    
    return magnitude_freq_components,frequency_components

def Inverse_Fourier_Transform(Magnitude_frequency_components):
    """
    Function to apply inverse fourier transform to transform the signal back to the time 
    domain
    
    After modifying the magnitude of the signal of some frequency components
    we apply the irfft to get the modified signal in the time domain (reconstruction)
    """
    
    Amplitude_time_domain = irfft(Magnitude_frequency_components) #Transform the signal back to the time domain.
    
    return np.real(Amplitude_time_domain)  #ensure the output is real.

#________________________________Modification of signals Function___________________________________#


def General_Signal_Equalization(SliderName, FrequencyMagnitude, FrequencyDomain, ValueOfSlider, ComponentRanges):
    """
    Function to apply changes in frequency / musical instrumntation  / vowels

        Parameters
        ----------
        SliderName            : According to mode Slider Name varies and its number example Vowels SliderName: A, E, U, T, S
        FrequencyMagnitude    : magnitude in frequency domain which you want to change it.
        FrequencyDomain       : frequency after apply fourier transform
        ValueOfSlider         : value to select the range of frequency to change magnitude.
        Componentranges       : ranges Component Frequency

        Return
        ----------
        FrequencyMagnitude : magnitude after apply changes.

    """

    for Name in range(len(ValueOfSlider)): #Loob on Slider exist in the selected mode
        if ValueOfSlider[Name]==None: # application by defalut set avlue of slider = none so we change it to 1
            ValueOfSlider[Name] = 1
        MagnitudeIndex = 0
        for Frequencies in FrequencyDomain: #Loob on components of frequencies(x-axis) in frequencyDomain
            if ComponentRanges[SliderName[Name]][1]> Frequencies and ComponentRanges[SliderName[Name]][0]<Frequencies : # Check if FreqeuncyDomain at location frequency example ate 1200 in ranages of the component frequnecy do the next
                FrequencyMagnitude[MagnitudeIndex] *= ValueOfSlider[MagnitudeIndex] #Modify the Magnitude of the frequencies
            MagnitudeIndex +=1
    
    return FrequencyMagnitude #return Modified Magnitude

def modify_medical_signal(Ecg_file, sliders_value):
    """
    Function to apply changes to a medical instrument signal.

    Parameters
    ----------
    Ecg_file       : CSV file of ECG 
        ECG file in CSV format.
    sliders_value  : list of float
        Values to be multiplied with the frequency components.

    Returns
    -------
    time_domain_amplitude : numpy array
        Time domain amplitude after applying changes.
    """
    fig1 = go.Figure()

    # Set x axis label
    fig1.update_xaxes(
        title_text="Frequency", 
        title_font={"size": 20},
        title_standoff=25
    )

    # Set y axis label
    fig1.update_yaxes(
        title_text="Amplitude (mv)",
        title_font={"size": 20},
        title_standoff=25
    )

    for i in range(len(sliders_value)):
        if sliders_value[i] is None:
            sliders_value[i] = 1

    # Get the Amplitude and Time from the CSV file
    time = Ecg_file.iloc[:, 0]
    amplitude = Ecg_file.iloc[:, 1]
    sample_period = time[1] - time[0]
    n_samples = len(time)

    # Apply FFT
    fourier = np.fft.fft(amplitude)
    frequencies = np.fft.fftfreq(n_samples, sample_period)
    counter = 0

    # Modify frequency components
    for value in frequencies:
        if value > 130:
            fourier[counter] *= sliders_value[0]
        if 130 >= value > 80:
            fourier[counter] *= sliders_value[1]
        if value <= 80:
            fourier[counter] *= sliders_value[2]
        counter += 1

    # Inverse FFT to get time domain amplitude
    time_domain_amplitude = np.real(np.fft.ifft(fourier))

    # Add scatter plot to figure
    fig_sig = fig1.add_scatter(x=time, y=time_domain_amplitude)

    # Show plot using Plotly
    pio.show(fig_sig)

    return time_domain_amplitude