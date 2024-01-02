import warnings


def warp_showwarning(message, category, filename, lineno, file=None, line=None):
    """Version of warnings.showwarning that always prints to sys.stdout."""
    msg = warnings.WarningMessage(message, category, filename, lineno, sys.stdout, line)
    warnings._showwarnmsg_impl(msg)


def warn(message, category=None, stacklevel=1):
    with warnings.catch_warnings():
        warnings.simplefilter("default")  # Change the filter in this process
        warnings.showwarning = warp_showwarning
        warnings.warn(message, category, stacklevel + 1)  # Increment stacklevel by 1 since we are in a wrapper
