import librosa
import os
import pandas as pd
import ast
import numpy as np

MAX_MEL_SHAPE = (647, 256)


class AudioHandler:
    def __init__(self, parent_dir, dataset='fma_small'):
        self.parent_dir = parent_dir
        self.dataset = dataset

    def gen_audio_df(self):
        audio_files = self.get_audio_files()
        audio_df = pd.DataFrame()

        for path, subdirs, files in audio_files:
            for file in files:
                audio_df = audio_df.append({'track_id': int(file[:-4]),
                                            'path': os.path.join(path, file)}, ignore_index=True)
        tracks = self.load_tracks()
        audio_df = audio_df.merge(tracks[['track_id', 'track_genre_top']], how='left', on='track_id')
        return audio_df

    def get_mel(self, path, n_fft=2048, n_mels=256, hop_length=1024):
        audio, sr = self.load_audio(path)
        mel = librosa.feature.melspectrogram(audio, sr=sr, n_fft=n_fft,
                                             n_mels=n_mels, hop_length=hop_length)
        return mel

    def load_many_mels(self, paths, n_fft=2048, n_mels=256, hop_length=1024):
        mels = []
        for path in paths:
            mel = self.get_mel(path, n_fft=n_fft,
                               n_mels=n_mels, hop_length=hop_length).T
            mel_shape = mel.shape
            if (mel_shape[0] < MAX_MEL_SHAPE[0]) or (mel_shape[1] < MAX_MEL_SHAPE[1]):
                pad_axis_0 = max(0, MAX_MEL_SHAPE[0] - mel_shape[0])
                pad_axis_1 = max(0, MAX_MEL_SHAPE[1] - mel_shape[1])
                mel = np.pad(mel, ((0, pad_axis_0), (0, pad_axis_1)),
                             mode='constant', constant_values=0)
            elif (mel_shape[0] > MAX_MEL_SHAPE[0]) or (mel_shape[1] > MAX_MEL_SHAPE[1]):
                print('Mel shape greater than expected', mel_shape)
                continue

            mels.append(mel)
        return np.array(mels)

    def load_tracks(self):
        tracks = pd.read_csv(self.parent_dir + 'fma_metadata/tracks.csv',
                             index_col=0, header=[0, 1])

        tracks_processed = pd.DataFrame()
        for lvl1, lvl2 in tracks.columns:
            tracks_processed[lvl1 + '_' + lvl2] = tracks[(lvl1, lvl2)]

        cols = [col for col in tracks_processed.columns if ('genres' in col) | ('tag' in col)]

        tracks_processed = tracks_processed.reset_index()

        for col in cols:
            tracks_processed[col] = tracks_processed[col].map(ast.literal_eval)

        cols = [col for col in tracks_processed.columns if ('date' in col) or ('year' in col)]

        for col in cols:
            tracks_processed[col] = pd.to_datetime(tracks_processed[col])

        return tracks_processed

    def get_audio_files(self):
        audio_files = os.walk(self.parent_dir + self.dataset)
        return audio_files

    def load_audio(self, path):
        audio, sr = librosa.load(path)
        return audio, sr
