import streamlit as st
import Equalizer_Functions
import pandas as pd

st.set_page_config(page_title="Equalizer", page_icon="✅", layout="wide")
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
    st.markdown('<h1 class="sidebar-title">Equalizer</h1>', unsafe_allow_html=True)
    file = st.file_uploader("Choose a file", type=["wav", "audio"], accept_multiple_files=False)
    st.markdown('<h2 class="sidebar-title">Choose the mode</h2>', unsafe_allow_html=True)
    Mode = st.selectbox(label="", options=[
                        'Uniform Range', 'Vowels', 'Musical Instruments', 'Biological Signal Abnormalities'])
    
#____________MAIN CODE______________#
if file:

    magnitude_at_time, sample_rate = Equalizer_Functions.to_librosa(file)
    Data_frame_of_medical = pd.DataFrame()
    pitch_step = 0
    spectogram = st.sidebar.checkbox(label="Spectogram")
    st.sidebar.write("## Audio before")
    st.sidebar.audio(file)
    if spectogram:
        st.session_state['Spectogram_Graph'] = True
    else:
        st.session_state['Spectogram_Graph'] = False

    if Mode == 'Uniform Range':
        dictnoary_values = {"0:1000": [0, 1000],
                            "1000:2000": [1000, 2000],
                            "3000:4000": [3000, 4000],
                            "4000:5000": [4000, 5000],
                            "5000:6000": [5000, 6000],
                            "6000:7000": [6000, 7000],
                            "7000:8000": [7000, 8000],
                            "8000:9000": [8000, 9000],
                            "9000:10000": [9000, 10000]}
        values_slider = []
        for key in dictnoary_values:
            range_start, range_end = dictnoary_values[key]
            num_steps = (range_end - range_start) // 100
            values_slider.append([range_start, range_end, num_steps])

    elif Mode == 'Vowels':
        vowels = ['E', 'T', 'A', 'O']
        dictnoary_values = {}
        for vowel in vowels:
            dictnoary_values[vowel] = [800, 1500] if vowel == 'E' else [700, 1800] if vowel == 'T' else [1000, 2500] if vowel == 'A' else [500, 2000]
        values_slider = [[0, 10, 1]] * len(vowels)

    elif Mode == 'Musical Instruments':
        dictnoary_values = {"Drum ": [0, 500],
                            "Flute": [500, 1000],
                            "Key": [1000, 2000],
                            "Piano": [2000, 5000]
                        }
        values_slider = []
        for value_range in dictnoary_values.values():
            start = value_range[0]
            end = value_range[1]
            step = (end - start) / 100
            values_slider.append([start, end, step])
    
    elif Mode == 'Biological Signal Abnormalities':
        dictnoary_values = {"Alpha Waves": [8, 13],
                            "Beta Waves": [13, 30],
                            "Gamma Waves": [30, 100],
                            "Delta Waves": [0.1, 4],
                            "Theta Waves": [4, 8]
                        }
        values_slider = []
        for freq_range in dictnoary_values.values():
            start = freq_range[0]
            end = freq_range[1]
            step = (end - start) / 100
            values_slider.append([start, end, step])
    
    Equalizer_Functions.processing_signal(Mode, list(dictnoary_values.keys()), values_slider, magnitude_at_time,
                              sample_rate, st.session_state['Spectogram_Graph'], dictnoary_values)