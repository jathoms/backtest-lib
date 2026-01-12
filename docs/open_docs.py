import pathlib
import sys
import webbrowser

open_flag = len(sys.argv) > 1 and sys.argv[1] == "--open"
output_path = sys.argv[2] if len(sys.argv) > 2 else ""

if open_flag:
    webbrowser.open(pathlib.Path(output_path, "index.html").resolve().as_uri())
