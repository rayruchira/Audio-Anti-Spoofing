import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import librosa as lr
import librosa.display
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


data_dir = './sounds/bonafide'
audio_files = glob(data_dir + '/*.flac')

n_fft = 2048
hop_length = 512
for ind in range(0,len(audio_files), 1):
    signal, sr = lr.load(audio_files[ind], sr=22050)

    MFCC = librosa.feature.mfcc(signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)

    window_size = 1024
    window = np.hanning(window_size)
    out = 2 * np.abs(MFCC) / np.sum(window)

    fig = plt.Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    p = librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max), ax=ax)
    fig.savefig('spoof/bonafide'+str(ind)+'.png')

