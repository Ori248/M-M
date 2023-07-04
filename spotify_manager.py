import spotipy
from spotipy.oauth2 import SpotifyOAuth
from credentials import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET
from constants import SPOTIFY_RETRIEVAL_LIMIT
from urllib.request import urlretrieve
from pathlib import Path


class SpotifyManager:
    """
    A manager of Spotify API usage. Used for authenticating the user, retrieving tracks from playlists,
    creating new playlists and putting the selected tracks in them.

    :attrs: redirect_uri - the HTTP server used for handling requests. Defaults to the local machine.
    :attrs: scopes - the Spotify API capabilities needed for the project
    :attrs: spotify - the main Spotify API object.
    :attrs: market - Spotify market to search in. Defaults to US due to great availability.
    """

    redirect_uri: str = r'http://localhost:8080'

    scopes: list = ['user-read-private', 'user-read-email', 'playlist-read-private',
                    'user-library-read', 'playlist-modify-public']

    spotify: spotipy.Spotify = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scopes, client_id=SPOTIFY_CLIENT_ID,
                                               client_secret=SPOTIFY_CLIENT_SECRET,
                                               redirect_uri=redirect_uri,
                                               show_dialog=False))
    market: str = 'US'

    def __init__(self):
        """
        Constructor. Retrieves the current session tokens, the user's id, and sets the destination playlist
        id to None.
        """
        auth_manager: SpotifyOAuth = self.spotify.auth_manager

        code = auth_manager.get_authorization_code()

        token = auth_manager.get_access_token(code=code, as_dict=False)

        user = self.spotify.current_user()

        self.user_id = user['id']

        self.token = token

        self.dst_playlist_id = None

    def get_playlist_items_and_preview_urls(self, playlist_id: str):
        """
        Generator that retrieves the items and their preview_urls from the playlist.

        :param: playlist_id: the source playlist id
        :return: a generator yielding an (item, preview_url) for each item in the playlist
        """

        num_tracks = self.get_playlist_length(playlist_id)

        for i in range(0, num_tracks, SPOTIFY_RETRIEVAL_LIMIT):
            results = self.spotify.playlist_items(playlist_id, market=self.market, offset=i,
                                                  limit=SPOTIFY_RETRIEVAL_LIMIT)

            for item in results['items']:
                preview_url = item['track'].get('preview_url')

                yield item, preview_url

    @staticmethod
    def download_preview(preview_url: str, dst_dir: Path, num: int) -> Path:
        """
        Downloads the given track preview and returns the destination of the download

        :param: preview_url: the track preview url
        :param: dst_dir: path to the download destination directory
        :param: num: the number according to which to name the downloaded .mp3 file
        :return: the downloaded file's path
        """

        download_dst = dst_dir / f'{num}.mp3'

        urlretrieve(preview_url, download_dst)

        return download_dst

    def create_playlist(self, playlist_name: str) -> str:
        """
        Creates a new playlist for the user with the given name

        :param: playlist_name: the new playlist's name
        :return: the new playlist's id
        """

        playlist = self.spotify.user_playlist_create(user=self.user_id, name=playlist_name)

        playlist_id = playlist["id"]

        self.dst_playlist_id = playlist_id

        return playlist_id

    def add_item_to_playlist(self, item: dict) -> dict:
        """
        Puts an item in the playlist

        :param: item: the item to add
        :return: the added item
        """

        self.spotify.playlist_add_items(playlist_id=self.dst_playlist_id, items=[item])

        return [item]

    def get_playlist_length(self, playlist_id: str) -> int:
        """
        Returns how many track the playlist with given id contains.
        :param: playlist_id: the id of the playlist
        :return: the playlist
        """

        playlist = self.spotify.playlist(playlist_id)

        return int(playlist['tracks']['total'])



