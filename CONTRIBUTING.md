# Contributing to SEMQuant

Thanks for considering a contribution!

## Development setup

```bash
git clone https://github.com/<your-org-or-user>/SEMQuant.git
cd SEMQuant
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev,ui]
```

## Run tests

```bash
pytest -q
```

## Style
This repo uses:
- `black` for formatting
- `ruff` for linting/import sorting

```bash
black .
ruff check .
```

## Pull requests
Please include:
- a short description of the change,
- before/after screenshots for UI changes,
- and a unit test for bug fixes when practical.
