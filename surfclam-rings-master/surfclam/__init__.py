"""surfclam - locate a surf clam shell and extract its growth lines.

Submodules are imported explicitly by callers (e.g. ``from surfclam import
imaging, locate``) rather than eagerly here, so a script that only needs the
light modules can run in an environment without scikit-image (e.g. the GPU
``sea`` env used for the deep models).
"""
