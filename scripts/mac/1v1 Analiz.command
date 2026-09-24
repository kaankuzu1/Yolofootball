#!/bin/bash
# Double-click in Finder to start the app. The first run sets everything up
# (a private Python environment in .venv, about 1-2 GB of downloads) and
# takes a few minutes; later runs start in seconds.
set -euo pipefail
cd "$(dirname "$0")/../.."

find_python() {
  for candidate in python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1 &&
       "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
      echo "$candidate"; return 0
    fi
  done
  return 1
}

if [ ! -x .venv/bin/python ]; then
  if ! PY=$(find_python); then
    echo "Python 3.11 veya daha yenisi gerekli."
    echo "Kurmak için: https://www.python.org/downloads/macos/  ya da  brew install python@3.12"
    read -r -p "Kapatmak için Enter'a basın." _
    exit 1
  fi
  echo "Python ortamı hazırlanıyor ($PY)…"
  "$PY" -m venv .venv
fi

if [ ! -f .venv/.app-installed ] || [ requirements-app.txt -nt .venv/.app-installed ] \
   || [ pyproject.toml -nt .venv/.app-installed ]; then
  echo "Gerekli paketler kuruluyor (ilk seferde birkaç dakika sürer)…"
  .venv/bin/python -m pip install --upgrade pip >/dev/null
  .venv/bin/python -m pip install -r requirements-app.txt
  .venv/bin/python -m pip install -e . >/dev/null
  touch .venv/.app-installed
fi

if [ ! -f assets/models/forzasys_soccer.pt ]; then
  echo "Uyarı: assets/models/forzasys_soccer.pt yok. Kamera açılır ama tespit ve analiz çalışmaz."
  echo "Model dosyalarını README'deki adımlarla assets/models/ klasörüne koyun."
fi

exec .venv/bin/python -m football_analysis.app "$@"
