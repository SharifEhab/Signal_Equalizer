#Import necessary libraries
import streamlit as st
import Equalizer_Functions
import pandas as pd
import soundfile as soundf
# Set page configuration
st.set_page_config(page_title="Equalizer", page_icon="✅", layout="wide")

# Open CSS file and add styling to page
with open("style.css") as design:
    st.markdown(f"<style>{design.read()}</style>", unsafe_allow_html=True)


#_________SESSION STATE INITIALIZATIONS________#

if 'Spectogram_Graph' not in st.session_state:
   st.session_state['Spectogram_Graph'] = False 
st.session_state.size1=0
st.session_state.flag=0
st.session_state.i=0
st.session_state.start=0

#__________SIDEBAR SECTION____________#

with st.sidebar:
# Add a title to the sidebar
    st.markdown('<h1 class="sidebar-title">Equalizer</h1>', unsafe_allow_html=True)
    # Create a file uploader in the sidebar and accept only .wav and .audio file types
    file = st.file_uploader("Choose a file", type=["wav", "audio","csv"], accept_multiple_files=False)
    # Add a title to choose the mode in the sidebar
    st.markdown('<h2 class="sidebar-title">Choose the mode</h2>', unsafe_allow_html=True)
    # Create a drop-down selector for choosing the mode in the sidebar
    Mode = st.selectbox(label="", options=[
    'Uniform Range', 'Vowels', 'Musical Instruments', 'Biological Signal Abnormalities'])
    
#____________MAIN CODE______________#
if file:
    if file.type == "audio/wav":

        magnitude_at_time, sample_rate,maximum_frequency = Equalizer_Functions.load_audio_file(file)
        Data_frame_of_medical = pd.DataFrame()
        spectogram = st.sidebar.checkbox(label="Spectogram")
        st.sidebar.write("## Audio before")
        st.sidebar.audio(file)
        if spectogram:
            st.session_state['Spectogram_Graph'] = True
        else:
            st.session_state['Spectogram_Graph'] = False

        # Depending on the selected equalizer mode, set the appropriate dictionary of frequency ranges and slider values
        #10 Equal ranges of frequency which changes dynamically according to the max freq in the input signal    
        if Mode == 'Uniform Range':
            dictnoary_values = {}
            frequency_step = maximum_frequency/10
            for frequency_range_index in range(10):
                start_freq = frequency_range_index*frequency_step
                end_freq = (frequency_range_index+1)*frequency_step
                freq_range = f"{int(start_freq)}:{int(end_freq)}"
                dictnoary_values[freq_range] = [start_freq,end_freq]
                
            values_slider = [[0,10,1]]*len(list(dictnoary_values.keys()))
            
            # For each frequency range, calculate the number of slider steps required and add it to the slider values list
            """
            for key in dictnoary_values:
                range_start, range_end = dictnoary_values[key]
                num_steps = (range_end - range_start) // 100
                values_slider.append([range_start, range_end, num_steps])
            """

        elif Mode == 'Vowels':
            # Create a list of vowels
            vowels = ['E', 'ʃ', 'A', 'ʊ','B']
            # Create an empty dictionary to store vowel ranges
            dictnoary_values = {}
            # Loop through each vowel
            for vowel in vowels:
                # Set the range of frequencies for each vowel based on its letter
                if vowel == 'E':
                    dictnoary_values[vowel] = [800, 1500]
                elif vowel == 'ʃ':
                    dictnoary_values[vowel] = [800,5000]
                elif vowel == 'A':
                    dictnoary_values[vowel] = [500,1200]
                elif vowel =='ʊ':
                    dictnoary_values[vowel] = [500, 2000]
                elif vowel == 'B':
                    dictnoary_values[vowel] = [1200,5000]
            # Create a list of slider values for each vowel
            values_slider = [[0, 10, 1]] * len(vowels)

        elif Mode == 'Musical Instruments':
            # Set the dictionary of frequency ranges for different musical instruments
            dictnoary_values = {"Tuba ": [40, 350],  # Frequency range for Tuba
                                "Picolo": [500, 3950]  # Frequency range for Piccolo
                                }

            # Set the list of slider values for each instrument
            # Here, we create a list with the same number of elements as the number of instruments in the dictionary
            # Each element in the list is itself a list with 3 values: start, end, and step size for the slider
            # We set the step size to 1 to allow for more precise adjustments of the sliders
            values_slider = [[0,10,1]]*len(list(dictnoary_values.keys()))
        Equalizer_Functions.processing_signal(Mode, list(dictnoary_values.keys()), values_slider, magnitude_at_time,
                              sample_rate, st.session_state['Spectogram_Graph'], dictnoary_values)
    elif file.type == "text/csv":        
        if Mode == 'Biological Signal Abnormalities':
                
            # create a dictionary containing the frequency ranges for various types of biological signal abnormalities
                dictnoary_values = {"arrythmia 1": [0, 0],
                                    "arrythmia 2": [0, 0],
                                    "arrythmia 3": [0, 0],
                                    }
                values_slider = [[0.0, 2.0, 1]] * \
                    len(list(dictnoary_values.keys()))
                Data_frame_of_medical = pd.read_csv(file)
                
                slider = Equalizer_Functions.generate_vertical_sliders(
                    list(dictnoary_values.keys()), values_slider, 0.1)
                Equalizer_Functions.modifiy_medical_signal(Data_frame_of_medical, slider)
                





    
    