#Import necessary libraries
import  streamlit_vertical_slider  as svs
from Signal_Generation_class import Signal
import numpy as np
import pandas as pd
import librosa
import librosa.display      
from numpy.fft import fft,rfft,rfftfreq,irfft,fftfreq
import plotly.graph_objects as go
import streamlit as st
import soundfile as soundf
import matplotlib.pyplot as plt
import time
import altair as alt
import plotly.graph_objs as go
from plotly.offline import iplot
import scipy.io.wavfile as wav 
#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ upload Function_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _#

#____End of functions/ variables for synthetic signal generation__________# 

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
   #array of arrays

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


def Get_Max_Frequency(amplitude_signal, sampling_rate):
    
    number_of_samples = len(amplitude_signal)
    
    sampling_period = 1/sampling_rate
    
    magnitude_freq_components = rfft(amplitude_signal)
    
    frequency_components = rfftfreq(number_of_samples,sampling_period)
    
    max_frequency = frequency_components[np.argmax(np.abs(magnitude_freq_components))]
    
    return max_frequency


def Inverse_Fourier_Transform(Magnitude_frequency_components):

    """
    Function to apply inverse fourier transform to transform the signal back to the time 
    domain
    
    After modifying the magnitude of the signal of some frequency components
    we apply the irfft to get the modified signal in the time domain (reconstruction)
    """
     
    Amplitude_time_domain = irfft(Magnitude_frequency_components) #Transform the signal back to the time domain.
    
    return np.real(Amplitude_time_domain)  #ensure the output is real.

#____Modification of signals Function_____#


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

    for Name in range(len(ValueOfSlider)):
        MagnitudeIndex = 0
        if ValueOfSlider[Name]==None: # application by defalut set avlue of slider = none so we change it to 1
            ValueOfSlider[Name] = 1
        for Frequencies in FrequencyDomain: #Loop on components of frequencies(x-axis) in frequencyDomain
            if Frequencies >ComponentRanges[SliderName[Name]][0] and Frequencies < ComponentRanges[SliderName[Name]][1] : # Check if frequency component of signal is within range of current slider 
                FrequencyMagnitude[MagnitudeIndex] *= ValueOfSlider[Name] #Modify the Magnitude of the frequencies
            MagnitudeIndex +=1
    
    return FrequencyMagnitude #return Modified Magnitude


def processing_signal(selected_mode,slider_labels,sliders_values,magnitude_signal_time,sampling_rate,bool_spectrogram,dict_freq_ranges,file):
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
      
    
    if selected_mode == 'Uniform Range' or 'Vowels' or 'Musical Instruments':
        col_timeplot_before,col_timeplot_after = st.columns(2)
    col_spectro_before,col_spectro_after = st.columns(2)
    all_sliders_values = generate_vertical_sliders(slider_labels,sliders_values)  #Selected values for each slider in an array
        
    magnitude_signal_frequency,frequency_components = Fourier_Transform_Signal(magnitude_signal_time,sampling_rate)
    
    magnitude_frequency_modified = General_Signal_Equalization(slider_labels,magnitude_signal_frequency,frequency_components,all_sliders_values,dict_freq_ranges)
    
    magnitude_time_modified = Inverse_Fourier_Transform(magnitude_frequency_modified)
    # original_audio(file)
    # modified_audio(magnitude_time_modified,sampling_rate)
    if selected_mode == 'Biological Signal Abnormalities' :
        modifiy_medical_signal(magnitude_signal_time,magnitude_time_modified,sampling_rate)   
          
    elif selected_mode == 'Uniform Range' or 'Vowels' or 'Musical Instruments' :
        with col_timeplot_before:
            show_plot(magnitude_signal_time,magnitude_time_modified,sampling_rate)   # Draw both original and modified plot in the time domain
            
    original_audio(file)
    modified_audio(magnitude_time_modified,sampling_rate)  
    if bool_spectrogram ==1:
        with col_spectro_before:
            Spectogram(magnitude_signal_time,"Before")
        with col_spectro_after:
            Spectogram(magnitude_time_modified,"After")
            


            
#____Aufio Before_____#
def original_audio(file):
    """
    This function displays the original audio file on the sidebar of a Streamlit app.

    Args:
        file (str): The path of the audio file.

    Returns:
        None
    """
    # Display a header for the original audio file section
    st.sidebar.write("## Audio before")
    
    # Display the audio file on the sidebar
    st.sidebar.audio(file)


#____Audio After____#

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
    st.sidebar.write("## Audio after")
    soundf.write("modified.wav",magnitude_time_modified,sample_rate) #saves the magnitude in time domain as an audio file named "output.wav" using the sample rate provided using the soundfile.write() function
    st.sidebar.audio("modified.wav")

def modifiy_medical_signal( amplitude,magnitude_time_modified,samplingrate):
    """
    Function to apply changes to a medical instrument signal.

    Parameters
    ----------
    Ecg_file       : wav file of ECG (PCG)
        ECG file in wav format.
    sliders_value  : list of float
        Values to be multiplied with the frequency components.

    Returns
    -------
    time_domain_amplitude : numpy array
        Time domain amplitude after applying changes.
    """
    
    #time_plot_col = st.columns(2)
    fig1 = go.Figure()

    # Set x axis label
    fig1.update_xaxes(
        title_text="Time", 
        title_font={"size": 20},
        title_standoff=25
    )
    
    # Set y axis label
    fig1.update_yaxes(
        title_text="Amplitude (mv)",
        title_font={"size": 20},
        title_standoff=25
    )
    
    time = np.arange(0,len(amplitude))/ samplingrate
    fig1.add_scatter(x=time, y=magnitude_time_modified)
    st.plotly_chart(fig1, use_container_width=True)

    return magnitude_time_modified

#____ Animation Function_____#

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
    #this function creates a animated line chart.
    #The chart displays two lines of data, one representing the amplitude of a signal over time
    #and the other representing the amplitude of the same signal after it has been processed. 
    brush = alt.selection_interval()
    chart1 = alt.Chart(df).mark_line().encode(
        x=alt.X('time', axis=alt.Axis(title='Time')),
    ).properties(
        width=330,
        height=170
    ).add_selection(
        brush).interactive()

    figure = chart1.encode(
        y=alt.Y('amplitude', axis=alt.Axis(title='Amplitude'))) | chart1.encode(
        y=alt.Y('amplitude after processing', axis=alt.Axis(title='Amplitude after'))).add_selection(
        brush)
    return figure

    
#____Plot Functions_____#

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

    return line_plot

def plotRep(df, size, start, num_of_element, line_plot):
    if 'current_state' not in st.session_state:
        st.session_state.current_state = start
    if 'step_df' not in st.session_state:
        st.session_state.step_df = df.iloc[st.session_state.current_state : st.session_state.current_state + size]
    # add buttons and slider to the sidebar
    is_playing = st.session_state.get('is_playing', True)
    if 'is_playing' not in st.session_state:
        st.session_state.is_playing = not is_playing
    button_col, button_col_2 = st.sidebar.columns(2)
    play_button = button_col.button('Play')
    pause_button = button_col_2.button('Pause')
    speed = st.sidebar.slider('Speed', min_value=1, max_value=50, value=25, step=1)

    if play_button:
        # st.session_state.play_pause_button_text = "⏸️"
        st.session_state.is_playing = True

    if pause_button:
        # st.session_state.play_pause_button_text = "▶️"
        st.session_state.is_playing = False

    if st.session_state.is_playing:
        i = st.session_state.current_state
        while i < num_of_element - size:
            step_df = df.iloc[i : size + i]
            st.session_state.step_df = step_df
            st.session_state.size1 = size + i
            lines = plot_animation(step_df)
            line_plot.altair_chart(lines)
            time.sleep(1/speed)
            if not st.session_state.is_playing:
                st.session_state.current_state = i
                break
            i += 1
            st.session_state.current_state = i
        if st.session_state.size1 == num_of_element - 1:
            st.session_state.is_playing = False
            step_df = df.iloc[0:num_of_element]
            lines = plot_animation(step_df)
            line_plot.altair_chart(lines)
            st.session_state.current_state = start
            st.session_state.step_df = df.iloc[start : start + size]
    else:
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


   
#___Spectogram Function____#

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
    D = librosa.stft(y)  # The result is a complex-valued matrix D that contains the magnitudes and phases of the frequency components of the signal at each time frame.
    
    # Convert the amplitude to logarithmic scale (decible)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # Set the figure size
    fig, ax = plt.subplots()
    
    # Plot the spectrogram
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax)
    
    # Set the title and axis labels
    ax.set(title=title_of_graph, xlabel='Time', ylabel='Frequency')
    
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    
    # Increase the font size of the title and axis labels
    ax.xaxis.label.set_fontsize(14)
    ax.yaxis.label.set_fontsize(14)
    ax.title.set_fontsize(16)
    
    # Increase the spacing between the subplots
    plt.subplots_adjust(hspace=0.6)
    
    # Display the plot in Streamlit
    st.pyplot(fig)