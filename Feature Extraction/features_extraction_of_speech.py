# import glob

import sklearn
import matplotlib.pyplot as plt
import librosa
import librosa.display


# data_dir = './heartbeat-sounds/set_b'
# audio_files = glob(data_dir + '/*.wav')

audio_path = './sounds/set_b/recorded_pkaudio.wav'
x , sr = librosa.load(audio_path)

# Wav form

plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)
# plt.show()


# Spectrogram form

X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
# plt.show()


#  MFCC form

plt.figure(figsize=(14,5))
mfccs = librosa.feature.mfcc(x, sr=sr)
print(mfccs.shape)
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
plt.show()


# Log-mel form

mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
print(mfccs.mean(axis=1))
print(mfccs.var(axis=1))
