import librosa
import numpy as np
import Signal_Analysis.features.signal
import matplotlib.pyplot as plt
import time
from typing import Union
from pathlib import Path
from hmmlearn import hmm
import pandas
import streamlit as st
import plotly.express as px
import scipy


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

def extract_feature_from_file(path_to_file: Union[str, Path]):
    plain_sig, fs = librosa.load(path_to_file, sr=None)
    trim_sig = np.trim_zeros(plain_sig, trim='fb')
    x = np.array_split(plain_sig, 50)
    zcr = [librosa.feature.zero_crossing_rate(i).mean(axis=1) for i in x]
    rms = [np.mean(i) for i in x]
    stft = librosa.amplitude_to_db(np.abs(librosa.stft(trim_sig, n_fft=1024)),ref=np.max)
    mfcc = librosa.feature.mfcc(y=plain_sig, sr=fs)
    spec_centroid = librosa.feature.spectral_centroid(y=trim_sig, sr=fs)
    spec_contrast = librosa.feature.spectral_contrast(y=trim_sig, sr=fs)


    return {'waveform': trim_sig, 'rms': rms, 'zcr': zcr, 'stft': stft, 'mfcc': librosa.power_to_db(mfcc, ref=np.max), 'spec_centroid': spec_centroid, 'spec_contrast': spec_contrast}

@st.cache_data
def injest():
    category_list = ['Effects', 'Human', 'Music', 'Nature', 'Urban']
    file_dict = {'Effects': {}, 'Human': {}, 'Music': {}, 'Nature': {}, 'Urban': {}}
    for category in file_dict.keys():
        for file_path in Path(f'MSoS_challenge_2018_Development_v1-00/Development/{category}').glob('*.wav'):
            data = extract_feature_from_file(file_path)
            file_dict[category][file_path.stem] = data
    return file_dict, category_list

def mfcc_develop(file_dict, category_list):
    hmm_models = []
    for category in category_list:
        label = category
        X = np.array([])
        y_words = []
        for key, value in file_dict[category].items():
            mfcc_features = value["mfcc"]
            if len(X) == 0:
                X = mfcc_features[:,:100]
            else:
                X = np.append(X, mfcc_features[:,:100], axis=0)            
            y_words.append(label)
        hmm_trainer = HMMTrainer()
        hmm_trainer.train(X)
        hmm_models.append((hmm_trainer, label))
        hmm_trainer = None
    return hmm_models

def mfcc_evaluate(hmm_models):
    eval_csv = pandas.read_csv('Logsheet_EvaluationMaster.csv', usecols=[0,2])
    correct_count = [0,0,0,0,0]

    for file_path in Path(f'Evaluation/').glob('*.wav'):
        plain_sig, fs = librosa.load(file_path)

        # Extract MFCC features
        mfcc_features = librosa.feature.mfcc(y=plain_sig, sr=fs)
        mfcc_features=mfcc_features[:,:100]

        scores=[]
        for item in hmm_models:
            hmm_model, label = item
            
            score = hmm_model.get_score(mfcc_features)
            scores.append(score)
        index=np.array(scores).argmax()
        matching_rows = eval_csv[eval_csv['File'] == file_path.name]
        category_for_file = matching_rows.iloc[0]['Category']
        if category_for_file == hmm_models[index][1]:
            correct_count[index] += 1
    return correct_count

data_dict, category_list = injest()

if st.button("Run HMM"):
    hmm_models = mfcc_develop(data_dict, category_list)
    count = mfcc_evaluate(hmm_models)
    st.write(count)

select_dev_category = st.selectbox(
    'Select category to inspect',
    (category_list))
select_file = st.selectbox('Select file to view', (data_dict[select_dev_category].keys()))

file_selected = data_dict[select_dev_category][select_file]
file_selected_keys = list(file_selected.keys())
st.audio(file_selected[file_selected_keys[0]],sample_rate=44100)
st.write(f"Available data: {[i for i in file_selected.keys()]}")

st.subheader("Waveform")
st.line_chart(file_selected[file_selected_keys[0]])

st.subheader("RMS binned")
st.line_chart(file_selected[file_selected_keys[1]])

st.subheader("ZCR")
st.line_chart(file_selected[file_selected_keys[2]])

st.subheader("STFT")
ff = px.imshow(file_selected[file_selected_keys[3]], aspect="auto")
st.plotly_chart(ff)
st.subheader("MFCC")
ff = px.imshow(file_selected[file_selected_keys[4]], aspect="auto")
st.plotly_chart(ff)