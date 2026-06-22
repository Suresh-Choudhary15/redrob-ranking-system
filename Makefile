test:
	pytest

requirements.txt:
    pip-compile pyproject.toml --output-file requirements.txt

requirements-platform.txt:
    pip-compile pyproject.toml --extra platform --output-file requirements-platform.txt

# precompute
# rank
# platform