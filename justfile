set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

typecheck:
	$fails = @() 
	just ty;             if ($LASTEXITCODE) { $fails += 'ty' } 
	just pyrefly;        if ($LASTEXITCODE) { $fails += 'pyrefly' } 
	just mypy;   		 if ($LASTEXITCODE) { $fails += 'mypy' } 
	just pyright;        if ($LASTEXITCODE) { $fails += 'pyright' }
	just ruffcheck;        if ($LASTEXITCODE) { $fails += 'ruffcheck' }


pyright: 
	uv run pyright 

mypy: 
	uv run mypy src

ty: 
	uv run ty check

pyrefly: 
	uv run pyrefly check

ruffcheck:
	uv run ruff check --fix

docs open="" output_dir="docs/_build/html":
  uv sync --group docs
  uv run sphinx-build -b html -W --keep-going "docs/source" "{{output_dir}}"
  @uv run python -c "import sys, pathlib, webbrowser; open_flag=sys.argv[1]; out=sys.argv[2]; (webbrowser.open(pathlib.Path(out, \"index.html\").resolve().as_uri()) if open_flag == \"--open\" else None)" "{{open}}" "{{output_dir}}"

doctest:
  uv sync --group docs
  uv run sphinx-build -b doctest -W --keep-going docs/source docs/_build/doctest
