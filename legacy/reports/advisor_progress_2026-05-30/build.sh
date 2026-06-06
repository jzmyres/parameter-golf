#!/usr/bin/env bash
# Build the advisor PDFs: the 1-page progress report and the comprehensive opg_doc note.
# Primary: tectonic (self-contained LaTeX engine, installed in a dedicated conda env).
# Fallbacks: a tectonic already on PATH, or a cluster TeXLive (pdflatex).
set -euo pipefail
cd "$(dirname "$0")"
SRCS=("progress_report.tex" "opg_doc.tex")

build_one() {
  local src="$1"
  if conda run --no-capture-output -n tex tectonic --version >/dev/null 2>&1; then
    conda run --no-capture-output -n tex tectonic "$src"
  elif command -v tectonic >/dev/null 2>&1; then
    tectonic "$src"
  elif command -v pdflatex >/dev/null 2>&1; then
    pdflatex -interaction=nonstopmode "$src" && pdflatex -interaction=nonstopmode "$src"
  else
    echo "No LaTeX engine found. Install tectonic:  conda create -y -n tex -c conda-forge tectonic" >&2
    exit 1
  fi
  echo "Built: $(pwd)/${src%.tex}.pdf"
}

for src in "${SRCS[@]}"; do
  build_one "$src"
done
