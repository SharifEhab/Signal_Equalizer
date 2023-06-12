# Signal Equalizer

A web application for manipulating signal frequencies and reconstructing the signal in real-time. The Signal Equalizer is a versatile tool designed for use in various domains, such as the music and speech industry, as well as biomedical applications like hearing aid abnormalities detection.

## Table of Contents
- [Introduction](#introduction)
- [Built with](#Built-with)
- [Description](#description)
- [Modes](#modes)
- [User Interface](#user-interface)
- [Signal Viewers](#signal-viewers)
- [Spectrograms](#spectrograms)
- [Authors](#Authors)
- [Acknowledgements](#Acknowledgements)

## Introduction
Signal equalization plays a fundamental role in the audio and speech industry, enabling users to adjust the magnitude of specific frequency components within a signal. This project aims to develop a web application that provides a user-friendly interface for manipulating signal frequencies through sliders and reconstructing the modified signal in real-time. Additionally, the application offers different modes to cater to specific needs, including uniform range mode, vowels mode, musical instruments mode, and biological signal abnormalities mode.

## Built with
<table>
  <tr>
    <td>
      <img src="https://img.shields.io/badge/programmig%20language-Python-red" alt="programming language">
    </td>
    <td>
      <img src="https://img.shields.io/badge/Framework-Streamlit-blue" alt="Framework">
    </td>
  </tr>
</table>

## Description
The Signal Equalizer web application allows users to open a signal and modify the magnitude of specific frequency components. The application provides a set of sliders that control the magnitude of frequency ranges or components within the signal. By adjusting these sliders, users can customize the output signal according to their preferences or specific requirements.

## Modes
The Signal Equalizer offers the following modes:

1. Uniform Range Mode: In this mode, the total frequency range of the input signal is divided uniformly into 10 equal ranges of frequencies. Each range is controlled by an individual slider in the user interface (UI).

2. Vowels Mode: This mode enables users to manipulate the magnitude of specific vowels within the signal. The sliders in this mode are designed based on research regarding the composition of speech vowels in terms of different frequency components/ranges.

3. Musical Instruments Mode: In this mode, users can adjust the magnitude of specific musical instruments present in the input music signal. Each slider corresponds to a particular musical instrument.

4. Biological Signal Abnormalities: This mode allows users to control the magnitude of specific abnormalities (e.g., ECG arrhythmia) within the input biological signal. Sliders are assigned to individual abnormalities for precise adjustment.

Users can easily switch between modes through an option menu or combobox in the UI. While the UI may undergo minor changes when switching modes (such as slider captions and their quantity), the overall layout remains consistent.

## User Interface
The Signal Equalizer's user interface consists of the following components:

- Sliders: The sliders provide control over the magnitude of specific frequency components or ranges within the signal. Users can adjust these sliders to modify the signal accordingly.

- Signal Viewers: The UI includes two signal viewers—one for the input signal and one for the output signal. Both viewers are synchronized to display the same time-part of the signal, ensuring a coherent representation. A play/stop/pause/speed-control panel allows users to control the playback of the signals.

- Spectrograms: Two spectrograms are provided—one for the input signal and one for the output signal. Any changes made to the equalizer sliders immediately reflect in the output spectrogram. Users have the option to toggle the visibility of the spectrograms.

The Signal Equalizer's UI takes into account boundary conditions, preventing scrolling beyond the signal's start or end points, as well as maintaining the signal within its minimum and maximum values.

## Signal Viewers
The Signal Equalizer includes two signal viewers—one for the input signal and one for the output signal. These viewers are synchronized, ensuring that they display the same time-part of the signal. Users can scroll or zoom on either viewer, and the other viewer will automatically adjust to maintain synchronization.

A play/stop/pause/speed-control panel is provided to facilitate the playback of the signals. Users can control the speed of playback and pause or resume the signal as needed.

## Spectrograms
The Signal Equalizer offers two spectrograms—one for the input signal and one for the output signal. These spectrograms provide a visual representation of the signal's frequency content.

Upon modifying any of the equalizer sliders, the output spectrogram immediately reflects the changes made. This allows users to visualize the impact of their adjustments on the signal's frequency distribution.

Users have the option to show or hide the spectrograms as per their preference.

## Authors

| Name | GitHub | LinkedIn |
| ---- | ------ | -------- |
| Omar Adel Hassan | [@Omar_Adel](https://github.com/omar-adel1) | [![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/omar-adel-59b707231/) |
| Sharif Ehab | [@Sharif_Ehab](https://github.com/SharifEhab) | [![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sharif-elmasry-b167a3252/) |
| Mostafa Khaled | [@Mostafa_Khaled](https://github.com/MostafaDarwish93) | [![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mostafa-darwish-75a29225b/) |
| Zyad Sowalim | [@Zyad_Sowalim](https://github.com/Zyadsowilam) | [![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/zyad-sowilam-798209228/) |

## Acknowledgements

Submitted to: Dr. Tamer Basha & Eng. Christina Adly

All rights reserved © 2023 to Team 4 - HealthCare Engineering and Management, Cairo University (Class 2025)

