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
import soundfile as soundf
import matplotlib.pyplot as plt
import time
import altair as alt

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

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ upload Function_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _#


def to_librosa(file_uploaded):
    """
        Function to upload file from librosa 
        Parameters
        ----------
        file uploaded 
        Return
        ----------
        y : samples
        sr : sampling rate      
    """
    if file_uploaded is not None:
        y, sr = librosa.load(file_uploaded)
        return y, sr
   
#________________________End of functions/ variables for synthetic signal generation__________________________________________________________________________________# 

def generate_vertical_sliders(array_slider_labels, array_slider_values,Slider_step=1):
    """
    Generate vertical sliders for different equalizer modes
    Parameters
    ----------
    array_slider_labels : label for each slider that controls the magnitude of certain frequency 
    ranges for the different modes 
    
    array_slider_values: factor that would be multiplied with the magnitude of some frequency 
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

    for Name in range(len(ValueOfSlider)): #Loop on Slider exist in the selected mode
        if ValueOfSlider[Name]==None: # application by defalut set avlue of slider = none so we change it to 1
            ValueOfSlider[Name] = 1
        MagnitudeIndex = 0
        for Frequencies in FrequencyDomain: #Loop on components of frequencies(x-axis) in frequencyDomain
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


def processing_signal(selected_mode,slider_labels,sliders_values,magnitude_signal_time,sampling_rate,bool_spectrogram,dict_freq_ranges):
    """
    Function to process the signal and show the time plots and spectrograms of uploaded signal before and after modifying the magnitude of some frequencies
    1- Perform fourier transform and get magnitude of freq components
    2- Modify magnitude of frequency components according to all_sliders_values
    3- Inverse fourier transform to get modified magnitude back in time domain
    Parameters
    ----------
    selected_mode : Uniform/music/vowels modes
    slider_labels : Label of each slider in the mode i.e.(Drums,Piano,etc..)
    magnitude_signal_time : Mag of signal in time domain
    bool_spectrogram : show or hide spectrogram
    dict_freq_ranges : dictionary containing labels and ranges of sliders
    
    Return
    ------
    
    """
 
    
    if selected_mode == 'Uniform Range' or 'Vowels' or 'Music Instrument' or 'Biological Signal Abnormalities':
        col_timeplot_before,col_timeplot_after = st.columns(2)
        col_spectro_before,col_spectro_after = st.columns(2)
        all_sliders_values = generate_vertical_sliders(slider_labels,sliders_values)  #Selected values for each slider in an array
        
        magnitude_signal_frequency,frequency_components = Fourier_Transform_Signal(magnitude_signal_time,sampling_rate)
        
        magnitude_frequency_modified = General_Signal_Equalization(slider_labels,magnitude_signal_frequency,frequency_components,all_sliders_values,dict_freq_ranges)
        
        magnitude_time_modified = Inverse_Fourier_Transform(magnitude_frequency_modified)
        
        modified_audio(magnitude_time_modified,sampling_rate)  
        
        with col_timeplot_before:
          
             show_plot(magnitude_signal_time,magnitude_time_modified,sampling_rate)   # Draw both original and modified plot in the time domain
        if bool_spectrogram ==1:
            with col_spectro_before:
               Spectogram(magnitude_signal_time,"Before")
            with col_spectro_after:
               Spectogram(magnitude_time_modified,"After")
                
#____________________________________Audio After______________________________________#

def modified_audio(magnitude_time_modified,sample_rate) :
    """
    Function to display audio after modifications
     Parameters
        ----------
        magnitude_time_modified : magnitude in time domain after modifications
        sample rate  
        Return
        ----------
        none            
    """            
    st.sidebar.write("#Audio after")
    soundf.write("modified.wav",magnitude_time_modified,sample_rate) #saves the magnitude in time domain as an audio file named "output.wav" using the sample rate provided using the soundfile.write() function
    st.sidebar.audio("modified.wav")

#______________________________ Animation Function_____________________________________#

def plot_animation(df):
    """
        Function to make the signal animated

        Parameters
        ----------
        df  : dataframe to be animated

        Return
        ----------
        figure             
    """ 
    brush = alt.selection_interval()
    chart1 = alt.Chart(df).mark_line().encode(
        x=alt.X('time', axis=alt.Axis(title='Time')),
    ).properties(
        width=400,
        height=200
    ).add_selection(
        brush).interactive()

    figure = chart1.encode(
        y=alt.Y('amplitude', axis=alt.Axis(title='Amplitude'))) | chart1.encode(
        y=alt.Y('amplitude after processing', axis=alt.Axis(title='Amplitude after'))).add_selection(
        brush)
    return figure

    
#______________________________Plot Functions_________________________________#

def currentState(df, size, num_of_element):
    """
    Function to display current state of dataframe

    Parameters
    ----------
    df            : Pandas dataframe
    size          : size of the dataframe
    num_of_element: number of elements to be displayed

    Return
    ----------
    chart         : chart of current state
    """

    if 'i' not in st.session_state:
        st.session_state.i = 0

    if st.session_state.i == 0:
        step_df = df.iloc[0:num_of_element]
    else:
        step_df = df.iloc[st.session_state.i: st.session_state.i + num_of_element]

    lines = plot_animation(step_df)
    line_plot = st.altair_chart(lines)

    if st.session_state.i + num_of_element < size:
        if st.button('Next'):
            st.session_state.i += num_of_element
    else:
        st.session_state.i = 0

    return line_plot


def plotRep(df, size, start, num_of_element, line_plot):
    if 'current_state' not in st.session_state:
        st.session_state.current_state = start
    if 'step_df' not in st.session_state:
        st.session_state.step_df = df.iloc[st.session_state.current_state : st.session_state.current_state + size]
    play_button = st.button('Play')
    pause_button = st.button('Pause')
    speed = st.slider('Speed', min_value=1, max_value=10, value=5)
    if play_button:
        st.session_state.flag = 0
    if pause_button:
        st.session_state.flag = 1
    if st.session_state.flag == 0:
        i = st.session_state.current_state
        while i < num_of_element - size:
            step_df = df.iloc[i : size + i]
            st.session_state.step_df = step_df
            st.session_state.size1 = size + i
            lines = plot_animation(step_df)
            line_plot.altair_chart(lines)
            time.sleep(1/speed)
            if st.session_state.flag == 1:
                # save the current state of the graph
                st.session_state.current_state = i
                break
            i += 1
            st.session_state.current_state = i
        if st.session_state.size1 == num_of_element - 1:
            st.session_state.flag = 1
            step_df = df.iloc[0:num_of_element]
            lines = plot_animation(step_df)
            line_plot.altair_chart(lines)
            # reset the current state to the start
            st.session_state.current_state = start
            st.session_state.step_df = df.iloc[start : start + size]
    else:
        # restore the current state of the graph
        lines = plot_animation(st.session_state.step_df)
        return line_plot.altair_chart(lines)

    return line_plot

def show_plot(samples, samples_after_moidifcation, sampling_rate):
    """
    Function to show plot

    Parameters
    ----------
    samples: ndarray
        Samples from librosa.
    samples_after_moidifcation: ndarray
        Samples after applying changes.
    sampling_rate: int
        Sampling rate from librosa.

    Return
    ----------
    None
    """
    time_before = np.arange(0, len(samples)) / sampling_rate
    time_after = np.arange(0, len(samples)) / sampling_rate

    df_after_upload = pd.DataFrame({
        'time': time_before[::500],
        'amplitude': samples[::500]
    })

    df_after_inverse = pd.DataFrame({
        'time_after': time_after[::500],
        'amplitude after processing': samples_after_moidifcation[::500]
    })

    common_df = pd.merge(df_after_upload, df_after_inverse, left_on='time', right_on='time_after')
    common_df.drop("time_after", axis=1, inplace=True)

    num_of_element = common_df.shape[0]  # number of elements in the dataframe
    burst = 10  # number of elements (months) to add to the plot
    size = burst
    line_plot = currentState(common_df, size, num_of_element)
    plotRep(common_df, size, st.session_state.start, num_of_element, line_plot)


   
#_______________________________Spectogram Function____________________________#


def Spectogram(y, title_of_graph):
    """
        Function to create a spectrogram of a given signal.

        Parameters
        ----------
        y : numpy array
            The time-domain signal to create the spectrogram of.
        title_of_graph : str
            The title to be displayed on the spectrogram plot.

        Return
        ----------
        None             
    """
    # Compute the STFT of the signal
    D = librosa.stft(y)
    
    # Convert the amplitude to decibels
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # Set the figure size
    fig, ax = plt.subplots(figsize=[10, 6])
    
    # Plot the spectrogram
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax)
    
    # Set the title and axis labels
    ax.set(title=title_of_graph, xlabel='Time', ylabel='Frequency')
    
    # Increase the font size of the title and axis labels
    plt.setp(ax.get_xticklabels(), fontsize=14)
    plt.setp(ax.get_yticklabels(), fontsize=14)
    ax.title.set_fontsize(16)
    
    # Increase the spacing between the subplots
    plt.subplots_adjust(hspace=0.6)
    
    # Display the plot in Streamlit
    st.pyplot(fig)

    
