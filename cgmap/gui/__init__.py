def launch_app(*args, **kwargs):
    from cgmap.gui.app import launch_app as _launch
    return _launch(*args, **kwargs)


__all__ = ['launch_app']
