"""Flatten the LaTeX report into pandoc-friendly LaTeX with numbers and refs resolved.

Usage: python3 flatten.py <report_dir> <out.tex>
Figure/table/section numbers and \\ref targets are taken from the LaTeX build (main.aux)
so the Word version matches the PDF.
"""
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
out = Path(sys.argv[2])

aux = (root / "main.aux").read_text()
labels = {m.group(1): m.group(2) for m in re.finditer(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", aux)}


def inline_inputs(text, base):
    def sub(m):
        name = m.group(1)
        if name == "figures/flow":
            return "FLOWIMG"
        p = base / (name if name.endswith(".tex") else name + ".tex")
        return inline_inputs(p.read_text(), base)
    return re.sub(r"\\input\{([^}]*)\}", sub, text)


def balanced(s, i):
    """Return (content, end) for the {...} group starting at s[i] == '{'."""
    assert s[i] == "{", s[i:i + 30]
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
    raise ValueError("unbalanced")


main = (root / "main.tex").read_text()
body = main[main.index("\\pagenumbering{arabic}") + len("\\pagenumbering{arabic}"):main.index("\\end{document}")]
body = inline_inputs(body, root)
body = body.replace("\\resizebox{\\linewidth}{!}{FLOWIMG}", "\\includegraphics{fig_flow.png}")
body = re.sub(r"\\begin\{align\*\}.*?\\end\{align\*\}", lambda m: "\n\n\\includegraphics{fig_model.png}\n\n", body, flags=re.S)

# --- \fig[width]{file}{caption} macro -------------------------------------------------
def expand_fig(s):
    res, i = [], 0
    while True:
        k = s.find("\\fig", i)
        if k < 0:
            res.append(s[i:]); break
        if s[k + 4:k + 5].isalpha():  # \figure etc.
            res.append(s[i:k + 4]); i = k + 4; continue
        res.append(s[i:k])
        j = k + 4
        if s[j] == "[":
            j = s.index("]", j) + 1
        f, j = balanced(s, j)
        cap, j = balanced(s, j)
        res.append("\\begin{figure}\\centering\\includegraphics{%s}\\caption{%s}\\label{fig:%s}\\end{figure}" % (f, cap, f))
        i = j
    return "".join(res)


body = expand_fig(body)


# --- figures with subfigures -> one paragraph of images + combined caption ---------------
def merge_subfigs(m):
    blk = m.group(0)
    subs = re.findall(r"\\begin\{subfigure\}\{[^}]*\}\\includegraphics\[[^\]]*\]\{([^}]*)\}\\caption\{([^}]*)\}\\end\{subfigure\}", blk)
    if not subs:
        return blk
    capm = re.search(r"\\end\{subfigure\}\s*\\caption\{", blk)
    cap, _ = balanced(blk, capm.end() - 1)
    imgs = "".join("\\includegraphics{%s}" % f for f, _ in subs)
    cap = cap.rstrip(".")
    if "(left)" not in cap:
        cap += " (left: %s; right: %s)" % (subs[0][1], subs[1][1])
    return "\\begin{figure}\\centering %s\\caption{%s.}\\end{figure}" % (imgs, cap)
body = re.sub(r"\\begin\{figure\}(\[H\])?.*?\\end\{figure\}", merge_subfigs, body, flags=re.S)

# --- boxed objective -------------------------------------------------------------------
lit = "\\fcolorbox{sasblue}{sasblue!6}{\\parbox{0.94\\linewidth}{"
k = body.find(lit)
if k >= 0:
    inner, end = balanced(body, k + len(lit) - 1)
    assert body[end] == "}", body[end:end + 10]
    body = body[:k] + "\\begin{quote}" + inner + "\\end{quote}" + body[end + 1:]
body = body.replace("\\begin{center}\n\\begin{quote}", "\\begin{quote}").replace("\\end{quote}\n\\end{center}", "\\end{quote}")

# --- listings -----------------------------------------------------------------------------
def listing(m):
    return "\\begin{verbatim}\n" + (root / m.group(1)).read_text() + "\n\\end{verbatim}"
body = re.sub(r"\\lstinputlisting\[[^\]]*\]\{([^}]*)\}", listing, body)

# --- tables: tabularx -> tabular with explicit widths -------------------------------------
def colspec(spec):
    spec = spec.replace("@{}", "").replace(" ", "")
    spec = re.sub(r"L\{([^}]*)\}", r"p{\1}", spec)
    spec = spec.replace("Y", "p{6cm}").replace("X", "p{5cm}")
    return spec
def tabx(m):
    spec, _ = balanced(m.string, m.end() - 1)
    return None
out_parts, i = [], 0
while True:
    k = body.find("\\begin{tabularx}{\\linewidth}", i)
    if k < 0:
        out_parts.append(body[i:]); break
    out_parts.append(body[i:k])
    j = k + len("\\begin{tabularx}{\\linewidth}")
    spec, j = balanced(body, j)
    out_parts.append("\\begin{tabular}{%s}" % colspec(spec))
    i = j
body = "".join(out_parts).replace("\\end{tabularx}", "\\end{tabular}")
body = re.sub(r"\\begin\{tabular\}\{([^}]*)\}", lambda m: "\\begin{tabular}{%s}" % m.group(1).replace("@{}", ""), body)
body = re.sub(r"\\multicolumn\{(\d+)\}\{@\{\}l\}", r"\\multicolumn{\1}{l}", body)
body = body.replace("\\addlinespace", "")
body = re.sub(r"\\setlength\{\\tabcolsep\}\{[^}]*\}", "", body)

# --- numbering of sections, figures, tables ----------------------------------------------
sec = sub = subsub = 0
appendix = False
fig = tab = 0
lines_out = []
env_stack = []
token_re = re.compile(r"\\(section|subsection|subsubsection)(\*?)\{|\\appendix|\\begin\{(figure|table|subtable|subfigure)\}|\\end\{(figure|table|subtable|subfigure)\}|\\caption\{")
res, i = [], 0
sub_letter = 0
for m in token_re.finditer(body):
    pass
s = body
res, i = [], 0
while True:
    m = token_re.search(s, i)
    if not m:
        res.append(s[i:]); break
    res.append(s[i:m.start()])
    tok = m.group(0)
    if tok == "\\appendix":
        appendix = True; sec = 0; i = m.end(); continue
    if m.group(1):
        kind, star = m.group(1), m.group(2)
        title, j = balanced(s, m.end() - 1)
        if star:
            res.append("\\%s*{%s}" % (kind, title))
        else:
            if kind == "section":
                sec += 1; sub = subsub = 0
                num = chr(64 + sec) if appendix else str(sec)
                if appendix:
                    title = "Appendix " + num + ": " + title
                    num = None
            elif kind == "subsection":
                sub += 1; subsub = 0
                num = "%s.%d" % (sec, sub)
            else:
                subsub += 1
                num = "%s.%d.%d" % (sec, sub, subsub)
            res.append("\\%s*{%s%s}" % (kind, (num + "\u00a0\u00a0") if num else "", title))
        i = j; continue
    if m.group(3):
        env_stack.append(m.group(3)); sub_letter_env = m.group(3)
        if m.group(3) in ("figure", "table"):
            sub_letter = 0
        res.append(tok); i = m.end(); continue
    if m.group(4):
        env_stack.pop(); res.append(tok); i = m.end(); continue
    # caption
    cap, j = balanced(s, m.end() - 1)
    env = env_stack[-1] if env_stack else None
    if env in ("subfigure", "subtable"):
        sub_letter += 1
        res.append("\\par\\textit{(%s) %s}\\par " % (chr(96 + sub_letter), cap))
    elif env == "figure":
        fig += 1; res.append("\\caption{Figure %d: %s}" % (fig, cap))
    elif env == "table":
        tab += 1; res.append("\\caption{Table %d: %s}" % (tab, cap))
    else:
        res.append("\\caption{%s}" % cap)
    i = j
body = "".join(res)

# subfigure/subtable wrappers -> plain content
body = re.sub(r"\\begin\{sub(figure|table)\}\{[^}]*\}", "", body)
body = re.sub(r"\\end\{sub(figure|table)\}(\\\\\[[^\]]*\])?", "", body)
body = body.replace("\\includegraphics[width=\\linewidth]{fig_", "\\includegraphics{fig_")


# --- blank lines inside tabulars break pandoc's parser ----------------------------------
body = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", lambda m: re.sub(r"\n\s*\n", "\n", m.group(0)), body, flags=re.S)

# --- simple inline math -> Unicode text (complex maths stays as Word equations) ----------
SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
REPL = [(r"\rightarrow", "→"), (r"\leq", "≤"), (r"\geq", "≥"), (r"\approx", "≈"), (r"\times", "×"),
        (r"^\dagger", "†"), (r"\rho", "ρ"), (r"\%", "%")]
def simple_math(m):
    t = m.group(1)
    u = t
    for a, b in REPL:
        u = u.replace(a, b)
    u = u.replace("\\in", "∈").replace("\\{", "{").replace("\\}", "}")
    u = re.sub(r"\b([A-Za-z])_\{?([a-z]+)\}?", lambda k: "\\emph{%s}\\textsubscript{%s}" % (k.group(1), k.group(2)), u)
    u = re.sub(r"\^\{(-?\d+)\}", lambda k: k.group(1).translate(SUP), u)
    chk = re.sub(r"\\(emph|textsubscript)\{[^}]*\}", "", u).replace("{0,1}", "")
    if "\\" in chk or "_" in chk or "^" in chk or "{" in chk:
        return m.group(0)
    u = re.sub(r"(?<![\d.])-(?=[\d.])", "−", u)
    u = u.replace("-", "−") if u.strip() == "-" else u
    # italicise single-letter variables like p, q, n, d
    u = re.sub(r"(?<![{a-z\\])\b([pqndkc])\b(?![}])", r"\\emph{\1}", u)
    return u.replace("%", "\\%")
body = re.sub(r"(?<!\\)\$([^$]+)\$", simple_math, body)
body = body.replace("∈ {0,1}", "∈ \\{0,1\\}")

# --- references ---------------------------------------------------------------------------
def ref(m):
    lab = m.group(1)
    if lab not in labels:
        print("WARN unresolved", lab); return "??"
    return labels[lab]
body = re.sub(r"\\ref\{([^}]*)\}", ref, body)
body = re.sub(r"\\label\{[^}]*\}", "", body)

# --- misc ----------------------------------------------------------------------------------
body = body.replace("\\clearpage", "").replace("\\bibliographystyle{plainnat}", "")
body = re.sub(r"\\bibliography\{[^}]*\}", lambda m: "\\section*{References}\n\n::REFS::\n", body)
body = body.replace("\\addcontentsline{toc}{section}{Executive Summary}", "")
body = body.replace("\\textperiodcentered{}", "·")

preamble = r"""\documentclass{article}
\usepackage{graphicx,booktabs,amsmath}
\graphicspath{{figures/}}
\begin{document}
"""
out.write_text(preamble + body + "\n\\end{document}\n")
print("figures", fig, "tables", tab)
