import librosa
import numpy as np
import Signal_Analysis.features.signal
import matplotlib.pyplot as plt
import time
from typing import Union
from pathlib import Path
from hmmlearn import hmm
import pandas as pd
import streamlit as st
import plotly.express as px
import scipy
from scipy.stats import kurtosis
from scipy.stats import skew, mode, iqr
from scipy.signal import coherence, hilbert

from PyOctaveBand.PyOctaveBand import octavefilter, _genfreqs

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size/2:]

def statistics(input):
    # This expects a 1D array 
    in_min = np.min(input)
    in_max = np.max(input)
    in_mean = np.mean(input)
    in_mode = mode(input)
    in_median = np.median(input)
    in_range = in_max - in_min
    in_variance = np.var(input)
    in_std = np.std(input)
    in_skew = skew(input)
    in_kurtosis = kurtosis(input)
    # in_entropy 
    in_percentiles = [np.percentile(input, n) for n in [10,25,50,75,90]]
    in_iqr = iqr(input)
    # in_autocorr = autocorr(input)
    return {"min": in_min, "max": in_max, "mean": in_mean, "mode": in_mode, "median": in_median, "range": in_range, "variance": in_variance,"std":in_std,"skew":in_skew,"kurtosis":in_kurtosis,"percentiles":in_percentiles,"iqr":in_iqr}

thirdoctfreqs = _genfreqs([20, 20000], fraction=3, fs=44100)[0]

PIANO_KEYS = [32.70, 34.65, 36.71, 38.89, 41.20, 43.65, 46.25, 49, 51.91, 55, 58.27, 61.74, 65.41, 69.3, 73.42, 77.78, 82.41, 87.31, 92.5, 98, 103.83, 110, 116.54, 123.47, 130.81, 138.59, 146.83, 155.56, 164.81, 174.61, 185, 196, 207.65, 220, 233.08, 246.94, 261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392, 415.3, 440, 466.16, 493.88, 523.25, 554.37, 587.33, 622.25, 659.25, 698.46, 739.99, 783.99, 830.61, 880, 932.33, 987.77, 1046.5, 1108.73, 1174.66, 1244.51, 1318.51, 1396.91,1479.98, 1567.98, 1661.22, 1760, 1864.66, 1975.53, 2093, 2217.46, 2349.32, 2489.02, 2637.02, 2793.83, 2959.96, 3135.96, 3322.44, 3520, 3729.31, 3951.07, 4186.01, 4434.92, 4698.63, 4978.03, 5274.04, 5587.65, 5919.91, 6271.93, 6644.88, 7040, 7458.62, 7902.13]

def extract_feature_from_file(path_to_file: Union[str, Path]):
    plain_sig, fs = librosa.load(path_to_file, sr=None)
    trim_sig = np.trim_zeros(plain_sig, trim='fb')
    x = np.array_split(plain_sig, 50)
    zcr = [librosa.feature.zero_crossing_rate(i).mean(axis=1) for i in x]
    rms = [np.mean(i) for i in x]
    stft = librosa.amplitude_to_db(np.abs(librosa.stft(trim_sig, n_fft=1024)),ref=np.max)
    mfcc = librosa.feature.mfcc(y=plain_sig, sr=fs)
    noise = np.random.normal(0,1,len(trim_sig))
    cohere = coherence(trim_sig,noise,fs)
    envelope = np.abs(hilbert(trim_sig))
    third_spl, third_freq = octavefilter(trim_sig, fs, fraction=3, order=4, limits=[20, 20000], show=0, sigbands=0)
    real_fft = np.abs(np.fft.rfft(trim_sig, 65536))
    fft_freqs = np.linspace(0, 22050, len(real_fft), True)
    harmonic_energy = 0
    anharmonic_energy = 0
    for frequency, value in zip(fft_freqs[:20000], real_fft[:20000]):
        for harmonic_freq in PIANO_KEYS:
            if 1.02*harmonic_freq > frequency > 0.98*harmonic_freq:
                harmonic_energy += value
            else:
                anharmonic_energy += value
    total_harmonic_energy = (harmonic_energy/np.sum(real_fft))*100
    high_freq_energy = np.sum(real_fft[20000:])/np.sum(real_fft)


    return {'waveform': {"plot": trim_sig, "stats": statistics(trim_sig)}, 'rms': {"plot": rms, "stats": statistics(rms)}, 'zcr': {"plot": zcr, "stats": statistics(zcr)}, 'stft': stft, 'mfcc': librosa.power_to_db(mfcc, ref=np.max), 'noiselike': {"plot": cohere[1], "stats": statistics(cohere[1])}, 'envelope': {"plot": envelope, "stats": statistics(envelope)},'third_octave':{'plot':third_spl, 'stats': statistics(third_spl)}, 'fft':{'plot':real_fft, 'stats': statistics(real_fft), 'harmonic_energy': total_harmonic_energy}}

@st.cache_data
def injest():
    category_list = ['Effects', 'Human', 'Music', 'Nature', 'Urban']
    file_dict = {'Effects': {}, 'Human': {}, 'Music': {}, 'Nature': {}, 'Urban': {}}
    for category in file_dict.keys():
        for file_path in Path(f'MSoS_challenge_2018_Development_v1-00/Development/{category}').glob('*.wav'):
            data = extract_feature_from_file(file_path)
            file_dict[category][file_path.stem] = data
    return file_dict, category_list

data_dict, category_list = injest()


select_dev_category = st.selectbox(
    'Select category to inspect',
    (category_list))
category_file_keys = (data_dict[select_dev_category].keys())
master_stats_dict = {'waveform': {"plot":[],"min":[], 'max':[],'mean':[], 'mode':[], 'median':[], 'range':[],'variance':[],'std':[],'skew':[],'kurtosis':[],'percentiles':[],'iqr':[]},'rms': {"plot":[],"min":[], 'max':[],'mean':[], 'mode':[], 'median':[], 'range':[],'variance':[],'std':[],'skew':[],'kurtosis':[],'percentiles':[],'iqr':[]},'zcr': {"plot":[],"min":[], 'max':[],'mean':[], 'mode':[], 'median':[], 'range':[],'variance':[],'std':[],'skew':[],'kurtosis':[],'percentiles':[],'iqr':[]},'noiselike': {"plot":[],"min":[], 'max':[],'mean':[], 'mode':[], 'median':[], 'range':[],'variance':[],'std':[],'skew':[],'kurtosis':[],'percentiles':[],'iqr':[]},'envelope': {"plot":[],"min":[], 'max':[],'mean':[], 'mode':[], 'median':[], 'range':[],'variance':[],'std':[],'skew':[],'kurtosis':[],'percentiles':[],'iqr':[]}, 'third_octave': {"plot":[],"min":[], 'max':[],'mean':[], 'mode':[], 'median':[], 'range':[],'variance':[],'std':[],'skew':[],'kurtosis':[],'percentiles':[],'iqr':[]}}


with st.expander("Macro-statistics"):
    col_array = st.columns(2)

    for column, data_type in enumerate(["waveform", "rms", "zcr", "noiselike", "envelope", "third_octave"]):
        for file in category_file_keys:
            file_selected = data_dict[select_dev_category][file]
            for method in ["plot","min", 'max', 'mean', 'mode', 'median', 'range','variance','std','skew','kurtosis','percentiles','iqr']:
                if method == "plot":
                    master_stats_dict[data_type][method].append([data_dict[select_dev_category][file][data_type]['plot']])
                else:
                    master_stats_dict[data_type][method].append(data_dict[select_dev_category][file][data_type]['stats'][method])
        with col_array[np.mod(column,2)]:
            st.subheader(data_type)
            for method in ['plot','mean', 'mode', 'median', 'range','variance','std','skew','kurtosis','percentiles','iqr']:
                if method != 'plot':
                    st.write(method)
                    try:
                        st.scatter_chart(master_stats_dict[data_type][method])
                    except TypeError as exc:
                        pass
                elif data_type == 'third_octave':
                    st.write(method)
                    chart = {f"y{counter}": response[0] for counter, response in enumerate(master_stats_dict[data_type][method])}
                    chart['x'] = thirdoctfreqs
                    chart = pd.DataFrame(chart)
                    try:
                        st.line_chart(chart, x='x')
                    except Exception as exc:
                        st.write(exc)
                        pass
                    meanchart = np.array(master_stats_dict[data_type][method]).reshape(30,300)
                    meanchart = np.mean(meanchart, axis=1)
                    meanchart = pd.DataFrame({'x': thirdoctfreqs, 'mean': meanchart})
                    st.line_chart(meanchart, x='x')

    harmonic_array = []
    for file in category_file_keys:
        file_selected = data_dict[select_dev_category][file]
        harmonic_array.append(file_selected['fft']['harmonic_energy'])
    st.write(np.mean(harmonic_array))
    st.write(np.min(harmonic_array))
    st.write(np.max(harmonic_array))



#min
#max
#mean
#mode
#median
#range
#variance
#std
#skew
#kurtosis
#percentiles
#iqr

select_file = st.selectbox('Select file to view', category_file_keys)

file_selected = data_dict[select_dev_category][select_file]
file_selected_keys = list(file_selected.keys())
# st.audio(file_selected[file_selected_keys[0]],sample_rate=44100)

with st.expander("Statistics"):
    st.subheader("Waveform")
    waveform_plot = file_selected[file_selected_keys[0]]["plot"]
    st.line_chart(waveform_plot)
    st.write(file_selected[file_selected_keys[0]]["stats"])


    st.subheader("RMS binned")
    rms_plot = file_selected[file_selected_keys[1]]["plot"]
    st.line_chart(rms_plot)
    st.write(file_selected[file_selected_keys[1]]["stats"])


    st.subheader("ZCR")
    zcr_plot = file_selected[file_selected_keys[2]]["plot"]
    st.line_chart(zcr_plot)
    st.write(file_selected[file_selected_keys[2]]["stats"])


    st.subheader("STFT")
    stft_plot = file_selected[file_selected_keys[3]]
    ff = px.imshow(stft_plot, aspect="auto")
    st.plotly_chart(ff)

    st.subheader("MFCC")
    mfcc_plot = file_selected[file_selected_keys[4]]
    ff = px.imshow(mfcc_plot, aspect="auto")
    st.plotly_chart(ff)

    st.subheader("Noise-like")
    noise_co_plot = file_selected[file_selected_keys[5]]["plot"]
    st.line_chart(noise_co_plot)
    st.write(file_selected[file_selected_keys[5]]["stats"])


    st.subheader("Envelope")
    envelope_plot = file_selected[file_selected_keys[6]]["plot"]
    st.line_chart(envelope_plot)
    st.write(file_selected[file_selected_keys[6]]["stats"])

    st.subheader("1/3 Octave frequency bands")
    third_plot = file_selected[file_selected_keys[7]]['plot']
    st.line_chart(third_plot)
    st.write(file_selected[file_selected_keys[7]]["stats"])

    st.subheader("FFT")
    fft_plot = file_selected[file_selected_keys[8]]['plot']
    fft_plot = pd.DataFrame({'x': np.linspace(0, 22050, len(fft_plot), True), 'y': fft_plot})
    st.line_chart(fft_plot, x='x')
    st.write(file_selected[file_selected_keys[8]]["harmonic_energy"])
    st.write(file_selected[file_selected_keys[8]]["stats"])