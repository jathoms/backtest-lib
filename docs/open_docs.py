import pathlib
import sys
import webbrowser

open_flag = sys.argv[1] == "--open"
output_path = sys.argv[2]

if open_flag:
    webbrowser.open(pathlib.Path(output_path, "index.html").resolve().as_uri())
