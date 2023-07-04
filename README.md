# M-M
This project, written entirely in Python, is a music mood classification project which uses a wrapper app that utilizes Spotify API to filter a playlist according to the desired moods.

A mel spectrogram is used to extract audio features from the music audio signals, as well as a Convolutional Neural Network (CNN) used for classification. The CNN classifies music according to 4 mood categories: happy, angry, sad and relaxed. This division follows Russell's model of Valence-Arousal from 1980 (link for more details: https://psycnet.apa.org/record/1981-25062-001). The prject also includes a data augmentation module whose purpose is to augment the dataset for better and more effective training of the model.

The librosa library is used to extract the audio features, and the Keras API of the TensorFlow library is used for crafting the model. The project also includes extensive use of the NumPy library, in addition to scikit-learn and spotipy, the Python wrapper for the Spotify web API.

The dataset used to train and evaluate the model is given in the following link: https://www.kaggle.com/datasets/blaler/turkish-music-emotion-dataset. Note that this dataset has been chosen for memory efficiency, and if memory is not an issue, the first dataset from the following link may well be used: http://mir.dei.uc.pt/downloads.html.

The model is evaluated using 5-fold cross-validation. In the best fold, the model achieved a test accuracy of 96.67%, as well as a very stable confusion matrix.

