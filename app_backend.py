from pathlib import Path
import os
from audio_feature_extraction import get_clip_mel_spectrogram
from constants import QUADRANTS_TO_MOODS, MEL_VALIDATION_SHAPE
from spotify_manager import SpotifyManager
import tensorflow as tf
import numpy as np
from exceptions import InvalidPlaylistLinkError, EmptyFieldError


def validate_user_choices(spotify: SpotifyManager, playlist_link: str, moods: list, new_playlist_name: str):
    """
    Validates the user's choices and raises errors if necessary
    :param: spotify: a SpotifyManager object
    :param: playlist_link: the link for the chosen playlist
    :param: moods: list of user-chosen moods
    :param: new_playlist_name: name of the new playlist
    """

    if not all([playlist_link, moods, new_playlist_name]):
        raise EmptyFieldError

    try:
        spotify.get_playlist_length(playlist_link)

    except Exception:
        raise InvalidPlaylistLinkError

    return 0


def filter_playlist(spotify: SpotifyManager, playlist_link: str, moods: list, new_playlist_name: str):
    """
    Filters the chosen playlist according to the chosen moods and puts the filtered items in the new playlist
    :param: spotify: a SpotifyManager object
    :param: playlist_link: the link for the chosen playlist
    :param: moods: list of user-chosen moods
    :param: new_playlist_name: name of the new playlist
    """

    model = tf.keras.models.load_model('model.h5')

    selected_moods = [mood.lower() for mood in moods]

    spotify.create_playlist(new_playlist_name)

    dst_dir = Path(r'./')

    playlist_items_and_preview_urls = spotify.get_playlist_items_and_preview_urls(playlist_id=playlist_link)

    for i, (item, preview_url) in enumerate(playlist_items_and_preview_urls):
        download_dst = spotify.download_preview(preview_url, dst_dir=dst_dir, num=i)

        mel_spectrogram = np.reshape(get_clip_mel_spectrogram(download_dst), MEL_VALIDATION_SHAPE)

        predictions = model.predict(mel_spectrogram, verbose=0)

        mood_decision = np.argmax(predictions)

        if QUADRANTS_TO_MOODS[mood_decision] in selected_moods:
            spotify.add_item_to_playlist(item['track']['uri'])

        os.remove(download_dst)

        yield i



