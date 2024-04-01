import time
import streamlit as st
import numpy as np
from pathlib import Path
import librosa
from scipy.stats import kurtosis
from scipy.stats import skew, mode, iqr
from scipy.signal import coherence, hilbert
import pandas as pd

from PyOctaveBand.PyOctaveBand import octavefilter, _genfreqs

PIANO_KEYS = [32.70, 34.65, 36.71, 38.89, 41.20, 43.65, 46.25, 49, 51.91, 55, 58.27, 61.74, 65.41, 69.3, 73.42, 77.78, 82.41, 87.31, 92.5, 98, 103.83, 110, 116.54, 123.47, 130.81, 138.59, 146.83, 155.56, 164.81, 174.61, 185, 196, 207.65, 220, 233.08, 246.94, 261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392, 415.3, 440, 466.16, 493.88, 523.25, 554.37, 587.33, 622.25, 659.25, 698.46, 739.99, 783.99, 830.61, 880, 932.33, 987.77, 1046.5, 1108.73, 1174.66, 1244.51, 1318.51, 1396.91,1479.98, 1567.98, 1661.22, 1760, 1864.66, 1975.53, 2093, 2217.46, 2349.32, 2489.02, 2637.02, 2793.83, 2959.96, 3135.96, 3322.44, 3520, 3729.31, 3951.07, 4186.01, 4434.92, 4698.63, 4978.03, 5274.04, 5587.65, 5919.91, 6271.93, 6644.88, 7040, 7458.62, 7902.13]

nl_mean_dict = {0.0006: [50/530, 100/530, 140/530, 120/530, 120/530], 0.007: [50/490, 100/490, 110/490, 120/490, 110/490], 0.0008: [0.2, 0.2, 0.3, 0.1, 0.2], 0.001: [20/65, 20/65, 5/65, 10/65, 10/65], 0.002: [90/ 155, 30/155, 5/155, 10/155, 20/155], 0.008:[70/116, 20/116, 1/116, 5/116, 20/116]}
nl_median_dict = {0.0004: [30/440, 80/440, 110/440, 110/440, 110/440], 0.00045: [50/350, 70/350, 90/350, 70/350, 70/350], 0.0005: [20/240, 50/240, 60/240, 60/240, 50/240], 0.0007: [50/170, 40/170, 30/170, 30/170, 20/170], 0.001: [50/110, 30/110, 10/110, 10/110, 10/110], 0.002: [90/143, 20/143, 10/143, 3/143, 20/143], 0.004: [15/41, 10/41, 1/41, 5/41, 10/41]}
nl_range_dict = {0.003: [40/420, 80/420, 110/420, 100/420, 90/420], 0.0035: [30/290, 60/290, 60/290, 80/290, 60/290], 0.004: [30/240, 50/240, 60/240, 50/240, 50/240], 0.005: [20/210, 50/210, 50/210, 50/210, 40/210], 0.008: [60/145, 30/145, 20/145, 15/145, 20/145], 0.01: [30/61, 10/61, 1/61, 10/61, 10/61], 0.04: [80/131, 20/131, 1/131, 10/131, 20/131]}
nl_iqr_dict = {0.0006: [30/310, 60/310, 80/310, 70/310, 70/310], 0.0007: [50/460, 80/460, 110/460, 110/460, 110/460], 0.0008: [20/260, 60/260, 70/260, 60/260, 50/260], 0.001: [20/140, 40/140, 30/140, 30/140, 20/140], 0.002: [90/170, 40/170, 10/170, 10/170, 20/170], 0.005: [70/120, 10/120, 10/120, 10/120, 20/120]}
to_var_dict = {40 : [30/98, 30/98, 1/98, 15/98, 22/98], 60: [20/118, 50/118, 8/118, 15/118, 25/118], 80: [30/115, 25/115, 5/115, 25/115, 30/115], 100: [25/115, 20/115, 10/115, 20/115, 40/115], 140: [30/210, 50/210, 20/210, 50/210, 60/210], 200: [30/200, 40/200, 30/200, 60/200, 40/200], 300: [50/250, 50/250, 40/250, 60/250, 50/250], 400: [40/160, 20/160, 60/160, 20/160, 20/160], 600: [15/160, 15/160, 90/160, 30/160, 10/160], 2000: [0.2, 0.2, 0.2, 0.2, 0.2]}
harmonic_dict = {30: [20/53, 3/53, 5/53, 10/53, 15/53], 40: [20/70, 20/70, 5/70, 10/70, 15/70], 50: [40/158, 40/158, 5/158, 33/158, 40/158], 60: [80/350, 80/350, 40/350, 70/350, 80/350], 70: [110/645, 160/645,110/645, 90/645, 175/645], 80: [20/160, 5/160, 120/160, 5/160, 10/260], 90: [10/43, 1/43, 30/43, 1/43, 1/43]}
zcr_median_dict = {0: [140/221, 40/221, 1/221, 10/221, 30/221], 0.01: [20/150, 30/150, 70/150, 20/150, 10/150], 0.02: [20/130, 10/130, 40/130, 30/130, 30/130], 0.03: [20/155, 20/155, 50/155, 25/155, 40/155], 0.04: [20/145, 45/145, 30/145, 20/145, 30/145], 0.05: [10/110, 30/110, 20/110, 20/110, 30/110], 0.07: [20/180, 50/180, 30/180, 50/180, 30/180], 0.1: [10/170, 50/170, 20/170, 50/170, 40/170], 0.2: [20/160, 40/160, 20/160, 40/160, 40/160], 0.7: [20/70, 10/70, 10/70, 20/70, 10/70]}



def injest(file, fs):
    trim_sig = np.trim_zeros(file, trim='fb')
    x = np.array_split(trim_sig, 50)
    zcr = [librosa.feature.zero_crossing_rate(i).mean(axis=1) for i in x]
    noise = np.random.normal(0,1,len(trim_sig))
    _, cohere = coherence(trim_sig,noise,fs)
    # envelope = np.abs(hilbert(trim_sig))
    # third_spl, third_freq = octavefilter(trim_sig, fs, fraction=3, order=4, limits=[20, 20000], show=0, sigbands=0)
    real_fft = np.abs(np.fft.rfft(trim_sig, 65536))
    fft_freqs = np.linspace(0, 22050, len(real_fft), True)
    harmonic_energy = 0
    for frequency, value in zip(fft_freqs[:16000], real_fft[:16000]):
        for harmonic_freq in PIANO_KEYS:
            if 1.02*harmonic_freq > frequency > 0.98*harmonic_freq:
                harmonic_energy += value
    total_harmonic_energy = (harmonic_energy/np.sum(real_fft))*100
    return [np.median(cohere), np.max(cohere) - np.min(cohere), total_harmonic_energy, np.median(zcr)]

filename = st.selectbox(label="Select file to classify", options=Path("./Evaluation").glob("*.wav"))


filename_glob = Path("./Evaluation").glob("*.wav")

filenames = list(filename_glob)


eval_dataset = pd.read_csv(Path("./Logsheet_EvaluationMaster.csv"), header=0, index_col=False, dtype=str)

index_to_cat = {0: 'Effects', 1: 'Human', 2: 'Music', 3: 'Nature', 4: 'Urban'}

best_weights = {0: []}
counter = 0
start_time = time.time()
for epoch in range(1000):
    rng = np.random.default_rng()
    counter+=1
    weights = rng.integers(low=1, high=1000, size=4)
    weights = np.divide(weights,[1000])
    total_correct = [0,0,0,0,0]
    # counter = 0
    # start_time = time.time()
    for filename in filenames:
        # counter += 1
        file_category = eval_dataset.loc[eval_dataset["File"] == filename.name]["Category"]
        file_category = str(file_category.values[0])

        loaded_file, Fs = librosa.load(path=f"./{filename}", sr=None)

        stats = injest(loaded_file, Fs)
        # st.write(file_category)

        probabilities = [0,0,0,0,0]
        method_counter = 0
        for stat, method in zip(stats, [nl_median_dict, nl_range_dict, harmonic_dict, zcr_median_dict]):
            probability_points = list(method.keys())
            if stat >= probability_points[-1]:
                probability = [0,0,0,0,0]
            elif stat <= probability_points[0]:
                probability = np.multiply(np.log(method[probability_points[0]]), [weights[method_counter]])
            else:
                for index, point in enumerate(probability_points[0:-2]):
                    if probability_points[index+1] > stat > point:
                        probability = np.multiply(np.log(method[probability_points[index+1]]), [weights[method_counter]])
                        break
            probabilities = np.add(probabilities,probability)
            method_counter+=1
        # st.write(probabilities)
        guessed_cat = index_to_cat[np.argmax(probabilities)]
        # st.write(guessed_cat)
        if guessed_cat == file_category:
            total_correct[np.argmax(probabilities)] += 1
        # print(counter)
        # print(total_correct)
        # print(np.sum(total_correct)/(counter))
    if not np.sum(total_correct) == 0:
        if np.sum(total_correct)/500 > list(best_weights.keys())[0]:
            best_weights = {(np.sum(total_correct)/500) : weights}


print("Done")
print(best_weights)
final_time = time.time() - start_time
print(final_time/60)
# st.write(final_time)
# st.write(total_correct)

