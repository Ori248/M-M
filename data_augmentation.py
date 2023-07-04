import numpy as np
from constants import SAMPLE_RATE
from scipy.io import wavfile
import os
from pathlib import Path
import librosa
import sys


def add_noise(audio_data: np.ndarray) -> np.ndarray:
    """
    :param: audio_data: audio data to which noise is added
    :return: an audio signal with noise
    """

    random = np.random.normal(0, 1, len(audio_data))

    return np.where(audio_data != 0.0, audio_data.astype('float64') + 0.02 * random, 0.0).astype(np.float32)


def time_shift(audio_data: np.ndarray, shift: int) -> np.ndarray:
    """
    :param: audio_data: audio data to which time shifting is applied
    :param: shift: the shifting rate, e.g how much every sample moves in time

    :return: a time shifted audio signal
    """

    return np.roll(audio_data, shift)


def data_augment(augmented_dataset_dst_dir: Path) -> int:
    """
    :param: augmented_dataset_dst_dir: path to the destination directory, where the augmented dataset will be saved.
    :return: 0 if successful, 1 if failed
    """

    for item in os.listdir(augmented_dataset_dst_dir):
        item_path = augmented_dataset_dst_dir / item

        if os.path.isdir(item_path):
            for audiofile in item_path.iterdir():
                audio_data, sr = librosa.load(audiofile)

                noised_audiofile = add_noise(audio_data)

                shifted_audiofile = time_shift(audio_data, shift=SAMPLE_RATE)

                audiofile_name = str(audiofile)[:-4]

                try:
                    wavfile.write(filename=f'{audiofile_name}_noised.wav', rate=SAMPLE_RATE, data=noised_audiofile)

                    print('Finished writing', f'{audiofile_name}_noised.wav')

                    wavfile.write(filename=f'{audiofile_name}_shifted.wav', rate=SAMPLE_RATE, data=shifted_audiofile)

                    print('Finished writing', f'{audiofile_name}_shifted.wav')

                except Exception as e:
                    sys.exit(f'An error occured: {e}')

    return 0




