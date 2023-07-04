import PySimpleGUIQt as sg
from spotify_manager import SpotifyManager
from app_backend import validate_user_choices, filter_playlist
from exceptions import InvalidPlaylistLinkError, EmptyFieldError


def filtering_window(spotify: SpotifyManager, playlist_link: str, moods: list, new_playlist_name: str):
    """
    Displays the playlist filtering window for the app
    :param: spotify: the SpotifyManager object initialized in main window
    :param: playlist_link: the link for the chosen playlist
    :param: moods: list of user-chosen moods
    :param: new_playlist_name: name of the new playlist
    """

    layout = [[sg.Text("Filtering your playlist...", key="text1")],
              [sg.ProgressBar(spotify.get_playlist_length(playlist_link), orientation='h', size=(20, 20),
                              key='PROGRESS-BAR')],
              [sg.Button('Start Filtering'), sg.Button("Stop Filtering"), sg.Exit(),
               sg.Button("Go back to the opening window")]]

    window = sg.Window("Filtering Playlist", layout, finalize=True)

    go_back_to_opening_window = False

    progress = filter_playlist(spotify, playlist_link, moods, new_playlist_name)

    run = True

    while run:
        event, values = window.read()

        if event in (sg.WIN_CLOSED, 'Exit', 'Stop Filtering', 'Go back to the opening window'):
            if event == 'Go back to the opening window':
                go_back_to_opening_window = True

                progress = filter_playlist(spotify, playlist_link, moods, new_playlist_name)

        if event == 'Start Filtering':
            for i in progress:
                event, values = window.read(timeout=1)

                window['PROGRESS-BAR'].UpdateBar(i + 1)

                if event in (sg.WIN_CLOSED, 'Exit', 'Stop Filtering', 'Go back to the opening window'):
                    if event == 'Go back to the opening window':
                        go_back_to_opening_window = True

                        progress = filter_playlist(spotify, playlist_link, moods, new_playlist_name)

                    run = False

                    break

    window.close()

    if go_back_to_opening_window:
        main()


def selection_window(spotify: SpotifyManager):
    """
    Displays the user option selection filtering window for the app
    :param: spotify: the SpotifyManager object initialized in main window
    """

    layout = [[sg.Text("Enter the playlist link:", key="text1")],
              [sg.Multiline(size=(40, 2), key='playlist_link')],
              [sg.Text("Choose the moods to filter the playlist according to", key="text2")],
              [sg.Listbox(values=["Happy", "Angry", "Sad", "Relaxed"], select_mode='multiple', key='moods',
                          size=(30, 3.2))],
              [sg.Text("Choose a name for the new playlist:", key="text3")],
              [sg.Multiline(size=(40, 2), key='new_playlist_name')],
              [sg.Button("Submit")]]

    window = sg.Window("User choices", layout, finalize=True)

    playlist_link = ''

    moods = []

    new_playlist_name = ''

    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED:
            break

        elif event == "Submit":
            playlist_link, moods, new_playlist_name = values.values()

            try:
                validate_user_choices(spotify=spotify, playlist_link=playlist_link, moods=moods,
                                      new_playlist_name=new_playlist_name)

                break

            except InvalidPlaylistLinkError:
                sg.Popup('Playlist link must be valid.')

            except EmptyFieldError:
                sg.Popup('All fields must be filled.')

    window.close()

    filtering_window(spotify=spotify, playlist_link=playlist_link, moods=moods, new_playlist_name=new_playlist_name)


def main():
    """
    Displays the initial window of the app
    """
    layout = [[sg.Text("Hello there! Press the button to start the app.")], [sg.Button("Login using Spotify")]]

    window = sg.Window("Welcome to M & M (Music & Mood)", layout)

    while True:
        event, values = window.read()

        if event == "Login using Spotify" or event == sg.WIN_CLOSED:
            spotify = SpotifyManager()

            break

    window.close()

    selection_window(spotify)
