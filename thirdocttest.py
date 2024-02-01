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

class HMMTrainer(object):
    def __init__(self, model_name='GaussianHMM', n_components=2, cov_type='full', n_iter=400):
        self.model_name = model_name
        self.n_components = n_components
        self.cov_type = cov_type
        self.n_iter = n_iter
        self.models = []

        if self.model_name == 'GaussianHMM':
            self.model = hmm.GaussianHMM(n_components=self.n_components, 
                    covariance_type=self.cov_type, n_iter=self.n_iter)
        else:
            raise TypeError('Invalid model type')

    # X is a 2D numpy array where each row is 13D
    def train(self, X):
        np.seterr(all='ignore')
        self.models.append(self.model.fit(X))

    def predict(self, X):
        print(self.model.predict(X))

    # Run the model on input data
    def get_score(self, input_data):
        return self.model.score(input_data)

thirdoctfreqs = _genfreqs([20, 20000], fraction=3, fs=44100)[0]

def extract_feature_from_file(path_to_file: Union[str, Path]):
    plain_sig, fs = librosa.load(path_to_file, sr=None)
    trim_sig = np.trim_zeros(plain_sig, trim='fb')
    x = np.array_split(plain_sig, 50)
    third_spl, third_freq = octavefilter(trim_sig, fs, fraction=3, order=4, limits=[20, 20000], show=0, sigbands=0)

    return {'third_octave':{'plot':third_spl}}

@st.cache_data
def injest():
    category_list = ['Effects', 'Human', 'Music', 'Nature', 'Urban']
    file_dict = {'Effects': {}, 'Human': {}, 'Music': {}, 'Nature': {}, 'Urban': {}}
    for category in file_dict.keys():
        for counter, file_path in enumerate(Path(f'MSoS_challenge_2018_Development_v1-00/Development/{category}').glob('*.wav')):
            if counter < 10:
                data = extract_feature_from_file(file_path)
                file_dict[category][file_path.stem] = data
    return file_dict, category_list

data_dict, category_list = injest()


select_dev_category = st.selectbox(
    'Select category to inspect',
    (category_list))
category_file_keys = (data_dict[select_dev_category].keys())
master_stats_dict = {'third_octave': {"plot":[]}}


with st.expander("Macro-statistics"):
    col_array = st.columns(2)

    for column, data_type in enumerate(["third_octave"]):
        for file in category_file_keys:
            file_selected = data_dict[select_dev_category][file]
            for method in ["plot"]:
                if method == "plot":
                    master_stats_dict[data_type][method].append([data_dict[select_dev_category][file][data_type]['plot']])
        with col_array[np.mod(column,2)]:
            st.subheader(data_type)
            for method in ['plot']:
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
                    meanchart = np.array(master_stats_dict[data_type][method]).reshape(30,10)
                    meanchart = np.mean(meanchart, axis=1)
                    meanchart = pd.DataFrame({'x': thirdoctfreqs, 'mean': meanchart})
                    st.line_chart(meanchart, x='x')