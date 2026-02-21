import librosa
import numpy as np

def preprocess_audio(file_path):

    audio, sr = librosa.load(file_path, sr=22050)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

    mfcc = mfcc.flatten()

    if len(mfcc) < 2376:
        mfcc = np.pad(mfcc, (0, 2376 - len(mfcc)))
    else:
        mfcc = mfcc[:2376]

    mfcc = np.expand_dims(mfcc, axis=-1)

    return mfcc
