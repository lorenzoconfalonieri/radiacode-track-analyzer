"""PyInstaller entry point: the same main() the `rcta` script runs."""
from rcta.gui.app import main

if __name__ == "__main__":
    raise SystemExit(main())
