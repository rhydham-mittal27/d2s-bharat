#!/usr/bin/env bash
# Build report/main.docx (Word version of the approach note) from the LaTeX sources.
# Needs: pdflatex + bibtex (run the LaTeX build first so main.aux is current), pandoc,
# poppler-utils (pdftoppm), python-docx. Usage: bash report/word/build_docx.sh
set -euo pipefail
W="$(cd "$(dirname "$0")" && pwd)"
R="$(dirname "$W")"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
cp -r "$R" "$T/report"
cd "$T/report"
# TikZ flow diagram and the RQ4 model as images (Word cannot take TikZ; equations stay crisp)
cp word/flow_standalone.tex word/model_standalone.tex .
python3 - <<'PY'
import re
s = open('sections/06_rq4.tex').read()
m = re.search(r"\\begin\{align\*\}.*?\\end\{align\*\}", s, re.S).group(0)
t = open('model_standalone.tex').read()
t = re.sub(r"\\begin\{document\}.*\\end\{document\}", lambda _: "\\begin{document}" + m + "\\end{document}", t, flags=re.S)
open('model_standalone.tex', 'w').write(t)
PY
pdflatex -interaction=nonstopmode flow_standalone >/dev/null
pdflatex -interaction=nonstopmode model_standalone >/dev/null
pdftoppm -png -r 250 -singlefile flow_standalone.pdf figures/fig_flow
pdftoppm -png -r 300 -singlefile model_standalone.pdf figures/fig_model
python3 word/flatten.py . "$T/flat.tex"
pandoc "$T/flat.tex" -f latex -t json -o "$T/flat.json"
python3 - "$T/flat.json" <<'PY'
import json, sys
p = sys.argv[1]; d = json.load(open(p))
is_ph = lambda b: b['t'] == 'Para' and any(x.get('t') == 'Str' and '::REFS::' in x.get('c', '') for x in b['c'])
d['blocks'] = [{'t': 'Div', 'c': [['refs', [], []], []]} if is_ph(b) else b for b in d['blocks']]
json.dump(d, open(p, 'w'))
PY
pandoc "$T/flat.json" -f json -t docx --citeproc --bibliography=references.bib \
  --csl=word/ieee.csl --resource-path=.:figures -o "$T/raw.docx"
python3 word/post.py "$T/raw.docx" "$R/main.docx"
