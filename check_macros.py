#!/usr/bin/env python3
"""check_macros.py -- guard against stale or hand-typed result numbers in the manuscript.

Checks
  1. Every result macro used in the manuscript is defined in generated_macros.tex with a real value
     (not the \\MacroMissing placeholder).
  2. generated_macros.tex has no duplicate definitions and no merge-conflict markers.
  3. No script other than generate_results_macros.py writes generated_macros.tex.
  4. Abstract, Introduction, Results and Discussion contain no hand-typed result numbers
     (decimals, percentages, N = ..., p-values). Design constants such as "30-minute" or
     "70--180 mg/dL" are allowed; add others to ALLOW below. Use --include-methods to lint Methods too.

Exit status 1 if anything is found (use in the pipeline after generate_results_macros.py).
"""
import argparse
import glob
import os
import re
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
OVERLEAF = os.path.join(BASE, "overleaf")
MACROS = os.path.join(OVERLEAF, "sections", "generated_macros.tex")
sys.path.insert(0, BASE)
from generate_results_macros import MACRO_PREFIXES  # noqa: E402

USE_PAT = re.compile(r"\\((?:" + "|".join(MACRO_PREFIXES) + r")[A-Za-z]*)")
NUM_PATS = [
    ("decimal", re.compile(r"(?<![A-Za-z\\{/_\-\d.])\d+\.\d+(?![\d.]*(?:\}|pt|cm|em|ex|mm|\\linewidth|\\textwidth))")),
    ("percentage", re.compile(r"(?<![A-Za-z\\{\d.])\d+(?:\.\d+)?\s*\\%")),
    ("sample size", re.compile(r"\b[Nn]\s*=\s*\$?\s*\d+")),
    ("p-value", re.compile(r"\bp\s*(?:<|=|>|\\leq?)\s*\$?\s*0?\.\d+")),
    ("count of N", re.compile(r"\b\d+\s+(?:evaluable\s+)?(?:participants|individuals|older adults|adults|configurations)\b")),
]
ALLOW = [
    r"95\\%", r"70--180", r"40--400", r"40,\s*400", r"\b0\.05\b", r"\b0\.001\b", r"\b0\.01\b", r"\\pm\s*5\\%", r"\\approx\s*0\.14",
    r"\b0\.10?\b", r"\b0\.20?\b", r"\b0\.30?\b", r"\b0\.50?\b", r"\b0\.8\b", r"\b1\.0\b",
    r"\b1\.5\b", r"\b2\.5\b", r"\b50\.0\b", r"\b18\.0182\b", r"3\.08", r"20\\%", r"50\\%",
    r"\b(?:three|four|two) ", r"5--15", r"30 to 60", r"300-second", r"10--14",
    r"3\.11\.2", r"2\.5\.1", r"1\.0\.6", r"2\.0\.1", r"3\.1\.1", r"3\.0\.5", r"1\.26\.4", r"1\.13\.1", r"2\.2\.2",
    r"\\linewidth", r"\\textwidth", r"width\s*=", r"1/n_", r"q = ", r"\\alpha", r"\\gamma",
    r"Weinstock 2016", r"sum to 100\\%", r"0\.5\s*(?:hr|h\b)",
]
SKIP_LINE = re.compile(r"\\csvreader|tabular=|table head|table foot|^\s*[A-Za-z.+\d]+=\\|\\includegraphics|\\label\{|\\cmidrule|\\multicolumn")


def strip_comment(line):
    out, i = [], 0
    while i < len(line):
        if line[i] == "%" and (i == 0 or line[i - 1] != "\\"):
            break
        out.append(line[i])
        i += 1
    return "".join(out)


def prose_lines(path, abstract_only=False):
    text = open(path, encoding="utf-8").read().splitlines()
    inside = not abstract_only
    for no, raw in enumerate(text, 1):
        if abstract_only:
            if "\\begin{abstract}" in raw:
                inside = True
            if "\\end{abstract}" in raw:
                inside = False
        if not inside:
            continue
        line = strip_comment(raw)
        if line.strip() and not SKIP_LINE.search(line):
            yield no, line


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--include-methods", action="store_true", default=True, help="Lint Methods section (enabled by default).")
    ap.add_argument("--no-methods", action="store_false", dest="include_methods", help="Skip linting Methods section.")
    ap.add_argument("--no-lint", action="store_true", help="Only check that macros are defined.")
    a = ap.parse_args()
    problems = []

    if not os.path.exists(MACROS):
        print("FAIL: generated_macros.tex does not exist; run generate_results_macros.py")
        return 1
    mtext = open(MACROS, encoding="utf-8").read()
    if re.search(r"^(<<<<<<<|=======|>>>>>>>)", mtext, re.M):
        problems.append("generated_macros.tex contains git merge-conflict markers; regenerate it")
    defs = re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$", mtext, re.M)
    names = [d[0] for d in defs]
    dup = sorted({n for n in names if names.count(n) > 1}) if len(names) != len(set(names)) else []
    if dup:
        problems.append(f"{len(dup)} macro(s) defined more than once: {', '.join(dup[:8])}")
    value = dict(defs)

    files = [(os.path.join(OVERLEAF, "main.tex"), True),
             (os.path.join(OVERLEAF, "sections", "introduction.tex"), False),
             (os.path.join(OVERLEAF, "sections", "results.tex"), False),
             (os.path.join(OVERLEAF, "sections", "discussion.tex"), False),
             (os.path.join(OVERLEAF, "sections", "supplement.tex"), False)]
    methods = (os.path.join(OVERLEAF, "sections", "methods.tex"), False)
    used = {}
    for path, abs_only in files + [methods]:
        if not os.path.exists(path):
            continue
        for no, line in prose_lines(path, False):
            for m in USE_PAT.findall(line):
                used.setdefault(m, (os.path.basename(path), no))
    undefined = sorted(m for m in used if m not in value)
    missing = sorted(m for m in used if value.get(m, "").strip() == "\\MacroMissing")
    for m in undefined:
        problems.append(f"undefined macro \\{m} (first used {used[m][0]}:{used[m][1]})")
    for m in missing:
        problems.append(f"macro \\{m} has no value yet, PDF shows ?? ({used[m][0]}:{used[m][1]})")

    writers = []
    for py in glob.glob(os.path.join(BASE, "*.py")) + glob.glob(os.path.join(BASE, "ExpMethods", "*.py")):
        if os.path.basename(py) in ("generate_results_macros.py", "check_macros.py"):
            continue
        src = open(py, encoding="utf-8", errors="replace").read()
        if re.search(r"generated_macros\.tex", src) and re.search(r"open\([^)]*[\"']w", src) and "newcommand" in src:
            writers.append(os.path.basename(py))
    if writers:
        problems.append("other scripts appear to write LaTeX macros: " + ", ".join(sorted(writers)))

    lint = []
    if not a.no_lint:
        allow = re.compile("|".join(ALLOW))
        bad_sign_pats = [
            ("negative reduction phrase", re.compile(r"\b(?:positive\s+mean\s+paired\s+reduction|reduction\s+of|reduced[^.\n]{0,40}\bby|improved[^.\n]{0,40}\bby|within)\s+(?:approximately\s+)?-\d", re.I)),
            ("negative sign after lower/higher", re.compile(r"\b(?:within)\s+[\$]?-?\d+(?:\.\d+)?\\%\s+(?:and|to)\s+[\$]?-\d", re.I)),
        ]
        expand_pat = re.compile(r"\\([A-Za-z]+)(?:\{\})?")
        def _expand_line(ln):
            return expand_pat.sub(lambda m: value.get(m.group(1), m.group(0)), ln)

        # Abstract expanded word count check (limit 150 words for npj Digital Medicine)
        abs_lines = [line for _, line in prose_lines(os.path.join(OVERLEAF, "main.tex"), abstract_only=True)]
        abs_raw = " ".join(l for l in abs_lines if "\\begin{abstract}" not in l and "\\end{abstract}" not in l)
        abs_exp = _expand_line(abs_raw)
        abs_exp = re.sub(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?", " ", abs_exp)
        abs_exp = abs_exp.replace("~", " ").replace("---", " ").replace("--", " ").replace("{", "").replace("}", "").replace("$", "")
        abs_words = [w for w in re.split(r"\s+", abs_exp.strip()) if w and not re.fullmatch(r"[,.;:()%]+", w)]
        if len(abs_words) > 150:
            problems.append(f"expanded abstract has {len(abs_words)} words (limit: 150)")

        for path, abs_only in files + ([methods] if a.include_methods else []):
            if not os.path.exists(path):
                continue
            for no, line in prose_lines(path, abs_only):
                exp = _expand_line(line)
                for skind, spat in bad_sign_pats:
                    if spat.search(exp):
                        lint.append(f"{os.path.basename(path)}:{no}: {skind} in expanded text: '{exp.strip()[:120]}'")
                masked = allow.sub(lambda m: " " * len(m.group(0)), line)
                masked = re.sub(r"\\cite\{[^}]*\}|\\ref\{[^}]*\}|\\eqref\{[^}]*\}", "", masked)
                for kind, pat in NUM_PATS:
                    for m in pat.finditer(masked):
                        ctx = masked[max(0, m.start() - 45): m.end() + 25].strip()
                        lint.append(f"{os.path.basename(path)}:{no}: hand-typed {kind} '{m.group(0).strip()}'  ...{ctx}...")

    print(f"{len(used)} result macros used in the manuscript; {len(value)} defined. (Abstract: {len(abs_words) if not a.no_lint else 'N/A'} words)")
    for p in problems:
        print("FAIL:", p)
    for l in lint:
        print("LINT:", l)
    if not problems and not lint:
        print("OK: every result number in the text is macro-driven and defined.")
    return 1 if (problems or lint) else 0


if __name__ == "__main__":
    sys.exit(main())
