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
