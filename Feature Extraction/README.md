## Audio Features
## Summary
* **Audio Signal:** An audio signal is a representation of sound which is caused by vibration and propagated by to and fro motion of a medium(generally air).
* The following are the important properties of an audio signal:
  * **Amplitude**: The maximum displacement of the particles. The highest point is called *crest* and lowest is called *trough*.
  * **Wavelength**: The distance between two successive crests or troughs.
![](https://cdn.analyticsvidhya.com/wp-content/uploads/2019/07/Waves-and-Communication-1.jpg)
  * **Frequency**: The no.of waves that pass over a fixed length in given time. It tells us about how fast signal is changing.
![](https://i.stack.imgur.com/y0qJV.png)

* There are two types of signals, **digital** and **analog**. Digital signals are discrete while analog signals are continuous and changing with time. Analog signals have infinite number of samples between any two intervals.
* Audio signal is an analog signal and hence hog a lot of memory and is very expensive computationally due to infinite number of samples.
* Hence **sampling** is done to convert audio signals from analog to digital.
![](https://cdn.analyticsvidhya.com/wp-content/uploads/2019/07/sampling.jpg.jpg)
* The **sampling rate** or **sampling frequency** is defined as the number of samples selected per second. 
### Feature Extraction
* **Time Domain**: Amplitudes are the features. They are recorded at different intervals of time. Can be visualized using amplitude-time graph. This completely ignores the rate of signal.

* **Frequency Domain**: Fourier Transform is a function that gets a signal in the time domain as input, and outputs its decomposition into frequencies. Audio signal is represented by amplitude as a function of frequency. It is a plot between frequency and amplitude. The features are the amplitudes recorded at different frequencies. This completely ignores the time domain and hence the sequence of the signal which is very important.

* **Spectrogram**: Spectrogram addresses the shortcomings of both time domain and frequency domain. It is a 2D plot between time and frequency. Each point is color coded and the intensity of the color tells us about the amplitude.

![](https://www.researchgate.net/profile/Achmad_Rizal4/publication/305183929/figure/fig1/AS:613869621895199@1523369331878/Bronchial-sounds-in-the-time-domain-the-frequency-domain-and-its-spectrogram-B-Signal.png)
* Spectrogram represents audio in the form of an image. This opens up a wide range of possibilities.
* Techniques used for images can now be used for audio as well given certain conditions are met.

### Mel Spectrum
* Mel comes from the word melody, to indicate that it is based on pitch comparisons. 
* Mel scale is the result of some non linear transformation of frequency scale.
* This Mel Scale is constructed such that sounds of equal distance from each other on the Mel Scale, also “sound” to humans as they are equal in distance from one another.
* For example on *Hertz* scale, the difference between 500Hz and 1000Hz is 500Hz and the difference between 9500Hz and 10,000Hz is also 500Hz but the difference noticeable by humans in the first case is significantly higher.
* Here is where mel comes in as distance on mel scale is proportionally differentiable by humans.
* A **frequency** of **1000 Hz equals 1000 mel**
![](https://in.mathworks.com/help/examples/audio/win64/ConvertBetweenMelScaleAndHzExample_01.png)

*   For shallower models Mel-frequency Cepstral Coefficients(MFCC) spectrogram can be used.

#### Click [here](http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/) for more information on MFCC.

