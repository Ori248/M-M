class InvalidPlaylistLinkError(Exception):
    """
    Raised when playlist link cannot be found
    """
    pass


class EmptyFieldError(Exception):
    """
    Raised when at least one of the fields is not filled
    """
    pass
