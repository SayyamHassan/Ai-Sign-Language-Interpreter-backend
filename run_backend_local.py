import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
VENV_SITE_PACKAGES = BASE_DIR / "venv" / "Lib" / "site-packages"

if VENV_SITE_PACKAGES.exists():
    sys.path.append(str(VENV_SITE_PACKAGES))

import uvicorn


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000)
