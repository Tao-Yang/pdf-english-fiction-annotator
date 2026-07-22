"""Entry point used to build the standalone Windows executable.

PyInstaller runs this module as ``__main__``; it simply forwards to the GUI's
``main`` so relative imports inside the ``annotator`` package resolve correctly.
"""

from annotator.gui import main

if __name__ == "__main__":
    raise SystemExit(main())
