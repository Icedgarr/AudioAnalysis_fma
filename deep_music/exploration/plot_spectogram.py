import librosa
import numpy as np
import matplotlib.pyplot as plt
from librosa import display


def plot_spectrogram(stf, mel=True):
    if mel:
        librosa.display.specshow(librosa.power_to_db(stf, ref=np.max),
                                 y_axis='mel', x_axis='time', sr=22050, hop_length=1024)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel spectrogram')
        plt.tight_layout()
        plt.show()
    else:
        librosa.display.specshow(librosa.amplitude_to_db(stf, ref=np.max),
                                 y_axis='log', x_axis='time', sr=22050, hop_length=1024)
        plt.title('Spectrogram')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.show()
