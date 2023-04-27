import streamlit as st
import Equalizer_Functions
import pandas as pd

st.set_page_config(page_title="Equalizer", page_icon="✅", layout="wide")
with open("style.css") as design:
    st.markdown(f"<style>{design.read()}</style>", unsafe_allow_html=True)

#________________________SESSION STATE INITIALIZATIONS_________________________#
if 'Spectogram_Graph' not in st.session_state:
   st.session_state['Spectogram_Graph'] = False 
#_____________________________SIDEBAR SECTION___________________________________#
with st.sidebar:
    st.markdown('<h1 class="sidebar-title">Equalizer</h1>', unsafe_allow_html=True)
    file = st.file_uploader("Choose a file", type=["csv", "wav", "audio/*"], accept_multiple_files=False)
    st.markdown('<h2 class="sidebar-title">Modes</h2>', unsafe_allow_html=True)
    Mode = st.radio(label="", options=[
                        'Uniform Range', 'Vowels', 'Musical Instruments', 'Biological Signal Abnormalities'])
    
#___________________________________MAIN CODE_________________________________________#
if file:
    if file.type == "audio/wav":
        SamplesMagnitude,SamplingRate=Equalizer_Functions.load_audio_file(file)
        spectogram = st.sidebar.checkbox(label="Spectogram")
        st.sidebar.write("## Audio before")
        st.sidebar.audio(file)
        if spectogram:
            st.session_state['Spectogram_Graph'] = True
        else:
            st.session_state['Spectogram_Graph'] = False

        if Mode == 'Uniform Range':
            Components = {         "0:1000": [0, 1000],
                                    "1000:2000": [1000, 2000],
                                    "3000:4000": [3000, 4000],
                                    "4000:5000": [4000, 5000],
                                    "5000:6000": [5000, 6000],
                                    "6000:7000": [6000, 7000],
                                    "7000:8000": [7000, 8000],
                                    "8000:9000": [8000, 9000],
                                    "9000:10000": [9000, 10000]
                                    }
            #SliderValues = [[0, 10, 1]]*len(list(Components.keys()))
            #slider_values = Equalizer_Functions.generate_vertical_sliders(list(Components.keys()), values_slider)

        elif Mode == 'Vowels':
            Components = {
                "E":[450, 2500],
                "T":[350, 2000],
                "A":[1000, 2500],
                "O":[500, 2000],
                "S":[4000,5500],
                "U":[250,1000]
            }
            
            
        elif Mode =="Musical Instrument":
            Components ={

            }

        SliderRanges = [[0, 10, 1]]*len(list(Components.keys()))
        
        Equalizer_Functions.processing_signal(Mode,list(Components.keys()),SliderRanges,SamplesMagnitude,SamplingRate,st.session_state['Spectogram_Graph'],Components)

    elif file.type =="csv":
        if Mode =='Biological Signal Abnormalities':
            Components ={

            }
            SliderRanges=[[0,4,1]]*len(list(Components.keys()))
            BiologicalSignal=pd.read_csv(file)
            
            Equalizer_Functions.modify_medical_signal(BiologicalSignal,Equalizer_Functions.generate_vertical_sliders(list(Components.keys())),SliderRanges,0.2)
            
