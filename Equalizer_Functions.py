import  streamlit_vertical_slider  as svs
from Signal_Generation_class import Signal
import numpy as np
import pandas as pd
import librosa
import librosa.display
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



def generate_slider(dict_values, values_slider):
    slider_values = []
    for i, (label, value_range) in enumerate(dict_values):
        slider_val = svs.vertical_slider(key=f"slider_{i}",min_value=value_range[0], max_value=value_range[1], step=values_slider[i][2])
        slider_values.append(slider_val)
    return slider_values


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

