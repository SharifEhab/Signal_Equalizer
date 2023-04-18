import streamlit as st
import Equalizer_Functions

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
    st.markdown('<h2 class="sidebar-title">Choose the mode</h2>', unsafe_allow_html=True)
    Mode = st.selectbox(label="", options=[
                        'Uniform Range', 'Vowels', 'Musical Instruments', 'Biological Signal Abnormalities'])
    
#___________________________________MAIN CODE_________________________________________#
if file:
    if file.type == "audio/wav":
        spectogram = st.sidebar.checkbox(label="Spectogram")
        st.sidebar.write("## Audio before")
        st.sidebar.audio(file)
        if spectogram:
            st.session_state['Spectogram_Graph'] = True
        else:
            st.session_state['Spectogram_Graph'] = False

        if Mode == 'Uniform Range':
            dict_values = [
                ("0:1000", [0, 1000]),
                ("1000:2000", [1000, 2000]),
                ("3000:4000", [3000, 4000]),
                ("4000:5000", [4000, 5000]),
                ("5000:6000", [5000, 6000]),
                ("6000:7000", [6000, 7000]),
                ("7000:8000", [7000, 8000]),
                ("8000:9000", [8000, 9000]),
                ("9000:10000", [9000, 10000])
            ]
            values_slider = [[0, 10, 1]]*len(dict_values)
            slider_values = Equalizer_Functions.generate_slider(dict_values, values_slider)

        elif Mode == 'Vowels':
            dict_values = [
                ("E",[800, 1500]),
                ("T",[700, 1800]),
                ("A",[1000, 2500]),
                ("O",[500, 2000])
            ]
            values_slider = [[0, 10, 1]]*len(dict_values)

        
