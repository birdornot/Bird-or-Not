# System
import os

# Data processing
import csv
import cv2
import numpy as np

# Machine learning
from tensorflow.keras.applications import VGG19

# Audio processing
import librosa

def BD(wav_files, model=None, M=None, S=None):
    """Bird detction"""
    print("Starting bird detection")
    print(wav_files)
    if model == None:
        model, M, S = load_BD_model()   # Trained model and Normalization factors (mean and std)
    size = 3 # Length of segments [seconds]

    if not isinstance(wav_files, list):
        wav_files = [wav_files]

# For each input audio file
    # wav_files = np.array(glob.glob(labelsPath + "/*.wav", recursive=True))
    for i, file in enumerate(wav_files):
        print("Detecting birds in ", file)
        try:
            fs = librosa.get_samplerate(file)
            segmented_audio, num_samples = segment(file, size)
            features = zip(*[extract_features(audio, fs, M, S) for audio in segmented_audio])
            print("Features extracted for ", file)

            print("Predictions being generated for ", file)
            preds = model.predict(np.vstack(features))
            baseline = baseline_gen(preds, num_samples, fs)
            export_baseline(baseline, file)
            print("Baseline exported for ", file)

        except Exception as exp:
            print(exp)

def load_BD_model():
    """Load detection model weights and normalization factors"""
    model = VGG19(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=(224,224,3),
        pooling=None,
        classes=2,
        classifier_activation="sigmoid",
    )

    # Load BD weights
    model.load_weights("BD_weights.h5")
    #model.summary()

    with open('BD_factors.npy', 'rb') as f:
        M, S = np.load(f, allow_pickle=True)

    return model, M, S

def segment(file, size):
    """Segment audio"""
    audio, fs = librosa.load(file, sr=None)
    overlap = int(size*fs/2)
    segmented_audio, num_samples = zip(*[(audio[i:i + size * fs],[i, i + size * fs])
                                         for i in range(0, len(audio) - overlap, overlap)])
    return segmented_audio, np.asarray(num_samples)

def prepare(mel_spec):
    """Resize spectrogram to fit the model"""
    new_array = cv2.resize(mel_spec, (224, 224)) # shape is (224,224)
    array_color = np.repeat(new_array[:, :, np.newaxis], 3, axis=2) # shape is (224,224,3) RGB convert
    array_with_batch_dim = np.expand_dims(array_color, axis=0) # shape is (1,224,224,3)
    return array_with_batch_dim

def normalize(mel_spec, M, S):
    """Normalize spectrogram"""
    mel_normalized = (mel_spec - M) / S
    return mel_normalized

def extract_features(audio, fs, M, S):
    """Extract mel features, normalize, and resize"""
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=fs,
                                             n_fft=2048,
                                             win_length= 1764,
                                             hop_length=882, # 882
                                             n_mels=80,
                                             center=False,
                                             fmin = 50,
                                             fmax = 12000)
    mel_normalized = normalize(mel_spec, M, S)
    feature = prepare(mel_normalized)
    return feature

def baseline_gen(preds, num_samples, fs):
    """Convert predictions to baseline labels"""
    bird_idx = np.where(preds[:, 0] > 0.9)[0]
    groups = [[bird_idx[0]]]
    for idx in bird_idx[1:]:
        if abs(idx - groups[-1][-1] <=2):
            groups[-1].append(idx)
        else:
            groups.append([idx])

    baseline = [[num_samples[min(x), 0]/fs, num_samples[max(x), 1]/fs, 'bird'] for x in groups]
    baseline[0][0] = max(baseline[0][0], np.finfo(float).eps)   # Convert zero to epsilon
    return baseline

def export_baseline(baseline, file):
    """Save baseline to input path"""
    file_name = os.path.splitext(os.path.basename(file))[0]
    print(file_name)
    dir_path = os.path.dirname(file)
    save_path = dir_path + '/BD Predictions/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    print(save_path)

    with open(save_path + file_name + '.txt', 'w') as f:
        wtrStats = csv.writer(f, delimiter="\t")
        wtrStats.writerows(baseline)

'''
# Import matlab weights and normalization info
import scipy 

weights = scipy.io.loadmat('weights.mat')
factors = scipy.io.loadmat('factors.mat')
M, S = factors['factors'][0]

# Set weights and biases
# Flatten net biases
for n in range(len(weights['weights'][0])):
    weights['weights'][0][n][0][1] = weights['weights'][0][n][0][1].flatten()

# Import untrained model 
from tensorflow.keras.applications import VGG19
model = VGG19(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=(224,224,3),
    pooling=None,
    classes=2,
    classifier_activation="softmax",
)
    
n = 0
for layer in model.layers:
    print(layer.name)
    if layer.get_weights() != []:
        layer.set_weights(weights['weights'][0][n][0])
        print(n)
        n=n+1

import numpy as np
model.save_weights("BD_weights.h5")
with open('BD_factors.npy', 'wb') as f:
    np.save(f, factors['factors'][0])
'''
