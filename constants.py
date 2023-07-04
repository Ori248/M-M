import numpy as np

QUADRANTS_TO_MOODS = {0: 'happy', 1: 'angry', 2: 'sad', 3: 'relaxed'}  # Maps numeric mood labels to text

MOODS_TO_QUADRANTS = {'happy': 0, 'angry': 1, 'sad': 2, 'relaxed': 3}  # Maps text mood labels to numeric ones

SPOTIFY_RETRIEVAL_LIMIT = 100  # Spotify API Limit

MEL_RESIZING_SHAPE = (400, 300)  # Resizing input for cv2

MEL_RESIZED_SHAPE = (300, 400)  # Resizing result from cv2

MEL_INPUT_SHAPE = MEL_RESIZED_SHAPE + (1,)  # Reshaped input for Keras CNN

MEL_VALIDATION_SHAPE = (1,) + MEL_INPUT_SHAPE  # Reshaped input for model real-life performance

SAMPLE_RATE = 22050  # In Hertz

AUDIO_CLIP_DURATION = 30  # In seconds

DATA_TYPE = np.float16  # For memory efficiency

TOTAL_NUM_CLIPS = 1200  # Total number of audio clips after data augmentation

N_SPLITS = 5  # Number of KFold splits





