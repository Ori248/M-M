import os
from joblib import dump, load
import numpy as np
from sklearn.preprocessing import StandardScaler
from audio_feature_extraction import get_clip_mel_spectrogram
from constants import MOODS_TO_QUADRANTS, DATA_TYPE, MEL_RESIZED_SHAPE, TOTAL_NUM_CLIPS
from pathlib import Path


def save_training_data(dataset_dir: Path) -> np.ndarray:
    """
    Extracts mel spectrograms from all audio samples and saves them into a destination .gz file.

    :param: dataset_dir: the source directory containing the data
    :return: mel spectrograms of all audio samples
    """

    mels = np.empty((TOTAL_NUM_CLIPS,) + MEL_RESIZED_SHAPE, dtype=DATA_TYPE)

    index = 0

    for i, path in enumerate(dataset_dir.iterdir()):
        if os.path.isdir(path):
            for audiofile in path.iterdir():
                mel = get_clip_mel_spectrogram(audiofile)

                print('Finished with', audiofile, 'index is', index)

                mels[index] = mel

                index += 1

    print('Saving data...')

    dump(mels, 'turkish_mels.gz', compress=True)

    return mels


def save_labels(dataset_dir: Path) -> np.array:
    """
    Extracts data labels from the given dataset and saves them into a .gz file.

    :param: dataset_dir: the source directory containing the data
    :return: labels for all audio samples
    """

    labels = np.zeros(TOTAL_NUM_CLIPS, dtype=np.int32)

    i = 0

    for item in os.listdir(dataset_dir):
        if os.path.isdir(dataset_dir / item):
            sub_dir_length = len(os.listdir(dataset_dir / item))

            label = MOODS_TO_QUADRANTS[item]

            labels[i:i+sub_dir_length] = np.repeat(label, sub_dir_length)

            print('label is', label)

            i += sub_dir_length

    dump(labels, 'turkish_labels.gz', compress=True)

    return labels


def normalize_data(data_file: Path) -> np.ndarray:
    """
    Normalizes the mel spectrograms data using StandardScaler(), squeezing the data into the range [-1, 1]
    and ensuring that it has 0 mean and 1 standard deviation.

    :param: data_file: a path to the file containing the mel spectrograms data
    :return: the normalized data
    """

    data = load(data_file)

    scaler = StandardScaler()

    flattened = data.flatten()

    normalized_data = scaler.fit_transform(flattened.reshape(-1, 1)).reshape((data.shape + (1,)))

    dump(normalized_data, 'turkish_normalized_mels.gz', compress=True)

    return normalized_data


