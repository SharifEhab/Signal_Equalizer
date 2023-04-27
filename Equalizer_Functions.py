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

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ upload Function_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _#

def to_librosa(file_uploaded):
    """
    Function that loads an audio file using librosa
    Parameters
    ----------
    file_uploaded : file object
        The uploaded file object to load into librosa
        
    Returns
    -------
    y : np.ndarray [shape=(n,) or (2, n)]
        The audio samples. If mono, then shape is (n,), otherwise (2, n)
    sr : number > 0 [scalar]
        The sample rate of the audio file.
    """
    # Check if a file was uploaded
    if file_uploaded is not None:
        # Load the file into librosa
        y, sr = librosa.load(file_uploaded)
        # Return the samples and sample rate
        return y, sr
   
#_________End of functions/ variables for synthetic signal generation___________________________# 

def generate_vertical_sliders(array_slider_labels, array_slider_values, Slider_step=1):
    """
    Generate vertical sliders for different equalizer modes
    
    Parameters:
        - array_slider_labels (list[str]): A list of labels for each slider that controls the magnitude of 
        certain frequency ranges for the different modes
        - array_slider_values (list[list[int]]): A list of values that will be used to create the sliders. 
        Each element of this list is a list of three integers representing the (Start, End, and Initial value) for 
        a slider.
        - Slider_step (int): The step of increment/decrement for each slider
    
    Returns:
        - different_slider_values (list[list[int]]): A list of lists, where each sublist has the (Start, End, and Step)
        values for each slider.
    
    """
    different_slider_values = []
    slider_columns = st.columns(len(array_slider_labels)) # Divide the page into equal columns to hold each slider
    for col_number in range(len(array_slider_labels)):
        with slider_columns[col_number]:
            current_slider = array_slider_values[col_number]
            # create a vertical slider using st.slider
            slider = st.slider(label="", key=array_slider_labels[col_number], 
                               min_value=current_slider[0], max_value=current_slider[1], 
                               value=current_slider[2], step=Slider_step, orientation='vertical')
            # Append the Start, End, and Step values of the slider to the different_slider_values list
            different_slider_values.append([current_slider[0], current_slider[1], slider])
            # Display the label of the slider
            st.write(array_slider_labels[col_number])
    return different_slider_values
       


def load_audio_file(path_file_upload):
    
    """
    Function to upload audio file given file path using librosa
    
    (Librosa is a Python package for analyzing and working with audio files,
    and it can handle a variety of audio file formats, including WAV, MP3, FLAC, OGG, 
    and many more.)
    
    Parameters:
    path_file_upload (str): Audio file path
    
    Output:
    audio_samples (numpy.ndarray): Audio samples
    sampling_rate (int): Sampling rate
    """
    
    # Check if file path is not None
    if path_file_upload is not None:
        
        # Load audio file and get audio samples and sampling rate using librosa
        audio_samples, sampling_rate = librosa.load(path_file_upload)
        
    # Return audio samples and sampling rate
    return audio_samples, sampling_rate  

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
    This function takes in an amplitude signal and its corresponding sampling rate, performs
    a fast Fourier transform on the signal to obtain the frequency components and their magnitudes.
    
    Parameters:
    amplitude_signal (array-like): An array-like sequence of amplitude values
    sampling_rate (int or float): The sampling rate of the signal, i.e., the number of samples per second
    
    Returns:
    magnitude_freq_components (array-like): An array-like sequence of magnitude values of the frequency components 
                                            obtained from the signal
    frequency_components (array-like): An array-like sequence of frequency values corresponding to the 
                                       frequency components obtained from the signal

    rrft-->  specialized version of the FFT algorithm that is 
    optimized for real-valued signals,  returns only positive frequencies
    
    rfftfreq--> function from the NumPy library is used to compute the frequencies 
    directly from the signal, returns an array of frequencies corresponding to the 
    output of the rfft function. 
    """
    # Get the number of samples in the signal
    number_of_samples = len(amplitude_signal)
    
    # Calculate the sampling period
    sampling_period = 1/sampling_rate
    
    # Perform a real-valued fast Fourier transform on the amplitude signal to get the frequency components and 
    # their magnitudes
    magnitude_freq_components = rfft(amplitude_signal)
    
    # Use the rfftfreq function to calculate the frequency components from the signal
    # The rfftfreq function returns an array of frequencies corresponding to the output of the rfft function
    frequency_components = rfftfreq(number_of_samples, sampling_period)
    
    # Return the magnitude and frequency components
    return magnitude_freq_components, frequency_components

def Inverse_Fourier_Transform(Magnitude_frequency_components):
    """
    Function to apply inverse fourier transform to transform the signal back to the time domain
    
    Args:
    Magnitude_frequency_components: numpy array containing the magnitude of frequency components of a signal
    
    Returns:
    numpy array containing the amplitude of the signal in the time domain after applying the inverse Fourier transform
    """
    
    # Apply inverse Fourier transform to get the signal in the time domain
    Amplitude_time_domain = irfft(Magnitude_frequency_components)
    
    # Ensure the output is real
    return np.real(Amplitude_time_domain)

#___________Modification of signals Function____________#


def General_Signal_Equalization(SliderName, FrequencyMagnitude, FrequencyDomain, ValueOfSlider, ComponentRanges):
    """
    Function to apply changes in frequency / musical instrumntation  / vowels

        Parameters
        ----------
        SliderName            : Name of the slider. Varies based on the mode. For example, for Vowels mode, SliderName: A, E, U, T, S
        FrequencyMagnitude    : Magnitude in frequency domain which needs to be changed.
        FrequencyDomain       : Frequency after applying Fourier Transform.
        ValueOfSlider         : Value to select the range of frequency to change magnitude.
        ComponentRanges       : Ranges Component Frequency.

        Return
        ----------
        FrequencyMagnitude : Magnitude after applying changes.

    """

    for Name in range(len(ValueOfSlider)): # Loop over each slider in the mode
        MagnitudeIndex = 0
        if ValueOfSlider[Name] is None: # If no slider value is selected, set it to 1
            ValueOfSlider[Name] = 1
        for Frequencies in FrequencyDomain: # Loop over each frequency component in FrequencyDomain
            if Frequencies > ComponentRanges[SliderName[Name]][0] and Frequencies < ComponentRanges[SliderName[Name]][1]: # Check if Frequencies falls in the range of the slider's frequency component
                FrequencyMagnitude[MagnitudeIndex] *= ValueOfSlider[Name] # Modify the magnitude of the frequency component
            MagnitudeIndex += 1
    
    return FrequencyMagnitude # Return the modified magnitude



def processing_signal(selected_mode, slider_labels, sliders_values, magnitude_signal_time, sampling_rate, bool_spectrogram, dict_freq_ranges):
    """
    Function to process the signal and show the time plots and spectrograms of uploaded signal before and after modifying the magnitude of some frequencies
    
    Parameters:
    -----------
    selected_mode : str
        The selected mode - either 'Uniform Range', 'Vowels', 'Musical Instruments', or 'Biological Signal Abnormalities'
    slider_labels : list of str
        Labels for each slider in the selected mode
    sliders_values : list of float
        Default values for each slider in the selected mode
    magnitude_signal_time : numpy.ndarray
        The magnitude of the signal in the time domain
    sampling_rate : int
        The sampling rate of the signal
    bool_spectrogram : bool
        Whether to show the spectrogram or not
    dict_freq_ranges : dict
        Dictionary containing the labels and frequency ranges for each slider
        
    Returns:
    --------
    None
    """
    
    # Check the selected mode and create column elements for before and after plots
    if selected_mode in ['Uniform Range', 'Vowels', 'Musical Instruments', 'Biological Signal Abnormalities']:
        col_timeplot_before, col_timeplot_after = st.columns(2)
        col_spectro_before, col_spectro_after = st.columns(2)
        
        # Generate vertical sliders for the selected mode
        all_sliders_values = generate_vertical_sliders(slider_labels, sliders_values)
        
        # Perform Fourier Transform and get magnitude of frequency components
        magnitude_signal_frequency, frequency_components = Fourier_Transform_Signal(magnitude_signal_time, sampling_rate)
        
        # Modify magnitude of frequency components according to slider values
        magnitude_frequency_modified = General_Signal_Equalization(slider_labels, magnitude_signal_frequency, frequency_components, all_sliders_values, dict_freq_ranges)
        
        # Perform Inverse Fourier Transform to get modified magnitude back in time domain
        magnitude_time_modified = Inverse_Fourier_Transform(magnitude_frequency_modified)
        
        # Play the modified audio
        modified_audio(magnitude_time_modified, sampling_rate)
        
        # Draw both original and modified plot in the time domain
        with col_timeplot_before:
            show_plot(magnitude_signal_time, magnitude_time_modified, sampling_rate)
        
        # Show the spectrogram of the signal before and after modification
        if bool_spectrogram == True:
            with col_spectro_before:
                Spectogram(magnitude_signal_time, "Before")
            with col_spectro_after:
                Spectogram(magnitude_time_modified, "After")

                
#_____________Audio After_____________#

def modified_audio(magnitude_time_modified, sample_rate):
    """
    Function to display audio after modifications.
    
    Parameters:
    -----------
    magnitude_time_modified : np.ndarray
        Magnitude in time domain after modifications.
    sample_rate : int
        Sample rate of the audio signal.
    
    Returns:
    --------
    None
    """
    st.sidebar.write("# Audio after")
    
    # Saves the magnitude in time domain as an audio file named "modified.wav" 
    # using the sample rate provided using the soundfile.write() function.
    soundf.write("modified.wav", magnitude_time_modified, sample_rate) 
    
    # Displays the audio file "modified.wav" using the st.sidebar.audio() function.
    st.sidebar.audio("modified.wav")


#__________ Animation Function_____________#

def plot_animation(df):
    """
        Function to make the signal animated

        Parameters
        ----------
        df  : dataframe to be animated

        Return
        ----------
        figure : an animated chart of the signal
    """ 

    # Create an interval selection brush
    brush = alt.selection_interval()

    # Create a line chart for the signal, with time on the x-axis
    chart1 = alt.Chart(df).mark_line().encode(
        x=alt.X('time', axis=alt.Axis(title='Time')),
    ).properties(
        width=400,
        height=200
    ).add_selection(
        brush).interactive()

    # Combine the signal chart with a second chart showing the signal after processing
    figure = chart1.encode(
        y=alt.Y('amplitude', axis=alt.Axis(title='Amplitude'))) | chart1.encode(
        y=alt.Y('amplitude after processing', axis=alt.Axis(title='Amplitude after'))).add_selection(
        brush)

    # Return the animated figure
    return figure


    
#___________Plot Functions____________#

def currentState(df, size, num_of_element):
    """
    Function to display current state of dataframe

    Parameters
    ----------
    df            : Pandas dataframe containing data to display
    size          : size of the dataframe
    num_of_element: number of elements to be displayed in each step

    Return
    ----------
    line_plot     : Altair chart of the current state
    """

    # Initialize the index if it doesn't exist in the session state
    if 'i' not in st.session_state:
        st.session_state.i = 0

    # Get the current step of the dataframe
    if st.session_state.i == 0:
        step_df = df.iloc[0:num_of_element]
    else:
        step_df = df.iloc[st.session_state.i: st.session_state.i + num_of_element]

    # Plot the current step of the dataframe
    lines = plot_animation(step_df)
    line_plot = st.altair_chart(lines)

    # If there are more elements to display, show a "Next" button
    if st.session_state.i + num_of_element < size:
        if st.button('Next'):
            st.session_state.i += num_of_element
    # Otherwise, reset the index to the beginning
    else:
        st.session_state.i = 0

    # Return the Altair chart of the current state
    return line_plot


def plotRep(df, size, start, num_of_element, line_plot):
    # Check if current_state exists in the session_state, if not set it to start
    if 'current_state' not in st.session_state:
        st.session_state.current_state = start
    # Check if step_df exists in the session_state, if not set it to a slice of the dataframe
    if 'step_df' not in st.session_state:
        st.session_state.step_df = df.iloc[st.session_state.current_state : st.session_state.current_state + size]
    # Create the play and pause buttons
    play_button = st.button('Play')
    pause_button = st.button('Pause')
    # Create the slider for controlling the animation speed
    speed = st.slider('Speed', min_value=1, max_value=10, value=5)
    # If the Play button is pressed, set the flag to 0
    if play_button:
        st.session_state.flag = 0
    # If the Pause button is pressed, set the flag to 1
    if pause_button:
        st.session_state.flag = 1
    # If the flag is 0 (i.e., if the animation is playing), run the animation loop
    if st.session_state.flag == 0:
        i = st.session_state.current_state
        # Loop through the dataframe and plot the animation step-by-step
        while i < num_of_element - size:
            # Slice the dataframe to get the next step
            step_df = df.iloc[i : size + i]
            # Update the step_df in the session_state
            st.session_state.step_df = step_df
            # Save the end index of the step_df in the session_state
            st.session_state.size1 = size + i
            # Plot the animation for the current step
            lines = plot_animation(step_df)
            line_plot.altair_chart(lines)
            # Wait for a short duration to slow down the animation
            time.sleep(1/speed)
            # If the Pause button is pressed, break out of the loop
            if st.session_state.flag == 1:
                # Save the current state of the graph
                st.session_state.current_state = i
                break
            # Move to the next step
            i += 1
            st.session_state.current_state = i
        # If we've reached the end of the dataframe, stop the animation and reset to the beginning
        if st.session_state.size1 == num_of_element - 1:
            st.session_state.flag = 1
            # Plot the last step of the animation
            step_df = df.iloc[0:num_of_element]
            lines = plot_animation(step_df)
            line_plot.altair_chart(lines)
            # Reset the current state to the start
            st.session_state.current_state = start
            st.session_state.step_df = df.iloc[start : start + size]
    else:
        # If the flag is 1 (i.e., if the animation is paused), restore the previous state
        lines = plot_animation(st.session_state.step_df)
        return line_plot.altair_chart(lines)

    # Return the line plot
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

    # Create arrays with time stamps for the samples before and after modification
    time_before = np.arange(0, len(samples)) / sampling_rate
    time_after = np.arange(0, len(samples)) / sampling_rate

    # Create dataframes with every 500th sample from the original and modified samples, respectively
    df_after_upload = pd.DataFrame({
        'time': time_before[::500],
        'amplitude': samples[::500]
    })
    df_after_inverse = pd.DataFrame({
        'time_after': time_after[::500],
        'amplitude after processing': samples_after_moidifcation[::500]
    })

    # Merge the dataframes based on the time column and drop the time_after column
    common_df = pd.merge(df_after_upload, df_after_inverse, left_on='time', right_on='time_after')
    common_df.drop("time_after", axis=1, inplace=True)

    # Set the initial burst size, line plot, and number of elements
    num_of_element = common_df.shape[0] # Number of elements in the dataframe
    burst = 10 # Number of elements (months) to add to the plot
    size = burst
    line_plot = currentState(common_df, size, num_of_element)

    # Call the function to plot the data
    plotRep(common_df, size, st.session_state.start, num_of_element, line_plot)



   
#__________Spectogram Function_________#

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

    # Compute the short-time Fourier transform (STFT) of the signal
    D = librosa.stft(y)
    
    # Convert the amplitude to decibels
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # Set the figure size and create a plot axis
    fig, ax = plt.subplots(figsize=[10, 6])
    
    # Plot the spectrogram
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax)
    
    # Set the title and axis labels
    ax.set(title=title_of_graph, xlabel='Time', ylabel='Frequency')
    
    # Increase the font size of the axis labels and title
    plt.setp(ax.get_xticklabels(), fontsize=14)
    plt.setp(ax.get_yticklabels(), fontsize=14)
    ax.title.set_fontsize(16)
    
    # Increase the spacing between the subplots
    plt.subplots_adjust(hspace=0.6)
    
    # Display the plot in Streamlit
    st.pyplot(fig)
