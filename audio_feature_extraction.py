import librosa
from librosa.feature import melspectrogram
import numpy as np
from constants import MEL_RESIZING_SHAPE, AUDIO_CLIP_DURATION, SAMPLE_RATE, DATA_TYPE
import cv2
from pathlib import Path


def extract_mel_spectrogram(audio_data: np.ndarray) -> np.ndarray:
    """

    :param: audio_data: audio data to analyze using melspectrogram
    :return: resized mel spectrogram for model training/testing/predicting

    """

    mel_spectrogam = melspectrogram(y=audio_data, sr=SAMPLE_RATE, dtype=DATA_TYPE)

    mel = librosa.amplitude_to_db(mel_spectrogam, ref=np.max)

    return cv2.resize(mel, MEL_RESIZING_SHAPE)


def pad_audio_data(audio_data: np.ndarray) -> np.ndarray:
    """

    :param: audio_data: audio data to adjust to length of AUDIO_CLIP_DURATION if necessary
    :return: padded/cut audio data of exact length AUDIO_CLIP_DURATION

    """

    data_len = len(audio_data)

    end = SAMPLE_RATE * AUDIO_CLIP_DURATION

    if end > data_len:
        audio_data = np.pad(audio_data, (0, end - data_len), 'mean')

    elif end < data_len:
        audio_data = audio_data[:end]

    return audio_data


def get_clip_mel_spectrogram(file_path: Path) -> np.ndarray:
    """
    Combination of the two above functions.

    :param: file_path: path to audio file to extract data from and analyze
    :return: extracted mel spectrogram from the file's audio data after adjusting to length AUDIO_CLIP_DURATION

    """

    audio_data, sample_rate = librosa.load(str(file_path), sr=SAMPLE_RATE)

    audio_data = pad_audio_data(audio_data)

    mel = extract_mel_spectrogram(audio_data)

    return mel


