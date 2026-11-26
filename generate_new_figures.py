#!/usr/bin/env python3
"""generate_new_figures.py -- Generate vector PDF figures for external validation, mechanism, and robustness.

Reads verified CSV outputs from:
  - external_results/{hall,dubosson}/*.csv
  - overleaf/tables/pool_mechanism_summary.csv
  - validation_results/validation_{cohort_sensitivity,gap_burden}.csv

Writes standalone pgfplots TeX sources compiled with pdflatex to:
  - overleaf/images/main/fig_external_validation.pdf
  - overleaf/images/main/fig_mechanism_context.pdf
  - overleaf/images/main/fig_validation_robustness.pdf
"""
import csv
import os
import shutil
import subprocess
import tempfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "overleaf", "images", "main")


def read_csv(rel_path):
    with open(os.path.join(BASE_DIR, rel_path), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compile_tex_pdf(tex_body: str, out_pdf: str):
    """Compile standalone pgfplots figure with pdflatex into exact-size vector PDF."""
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tex_path = os.path.join(tmp, "fig.tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(tex_body)
        for _ in range(2):
            res = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "fig.tex"],
                cwd=tmp,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if res.returncode != 0:
                raise RuntimeError(f"pdflatex failed for {out_pdf}:\n{res.stdout[-2500:]}")
        shutil.copyfile(os.path.join(tmp, "fig.pdf"), out_pdf)
    print(f"Wrote {os.path.relpath(out_pdf, BASE_DIR)}")


COMMON_PREAMBLE = r"""\documentclass{article}
\usepackage[T1]{fontenc}
\usepackage{amsmath}
\usepackage{lmodern}
\usepackage{xcolor}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\definecolor{hbmlred}{HTML}{C0392B}
\definecolor{hbmlblue}{HTML}{1F5A8A}
\definecolor{hbmlteal}{HTML}{1B7F79}
\definecolor{hbmlgreen}{HTML}{2E8B57}
\definecolor{hbmlorange}{HTML}{D97706}
\definecolor{hbmlgrey}{HTML}{6B7280}
\pagestyle{empty}
"""


def build_external_validation_fig():
    hall_stats = read_csv("external_results/hall/statistical_tests_best_expert.csv")
    hall_mfar = read_csv("external_results/hall/clinical_hypo_matched_far.csv")

    ctx_order = {"6": 0, "12": 1, "24": 2, "full": 3}
    hall_stats.sort(key=lambda r: (float(r["horizon"]), ctx_order[r["context"]]))

    row_labels = []
    bars90 = []
    bars95 = []
    sig_pts = []
    nonsig_pts = []
    n_rows = len(hall_stats)
    for idx, r in enumerate(hall_stats):
        y = n_rows - idx
        hz = "0.5h" if float(r["horizon"]) == 0.5 else f"{int(float(r['horizon']))}h"
        ctx = "full" if r["context"] == "full" else f"{r['context']}h"
        exp = "Mean" if r["expert"] == "RunningMean" else r["expert"]
        # Wrap label in braces so PGFPlots treats the internal comma/dot as part of one tick label
        row_labels.append("{" + f"{hz} / {ctx} ({exp})" + "}")
        mean = float(r["paired_diff_mean"])
        lo95, hi95 = float(r["ci95_mean_low"]), float(r["ci95_mean_high"])
        lo90, hi90 = float(r["tost_boot_ci90_low"]), float(r["tost_boot_ci90_high"])
        pbh = float(r["wilcoxon_p_bh"])
        bars95.append(rf"\draw[hbmlblue!70, line width=0.8pt] (axis cs:{lo95:.4f},{y}) -- (axis cs:{hi95:.4f},{y});")
        bars90.append(rf"\draw[hbmlblue, line width=2.4pt] (axis cs:{lo90:.4f},{y}) -- (axis cs:{hi90:.4f},{y});")
        if pbh < 0.05:
            sig_pts.append(f"({mean:.4f},{y})")
        else:
            nonsig_pts.append(f"({mean:.4f},{y})")

    curves = {m: [] for m in ("HBML", "BestExpert", "Persistence")}
    for r in hall_mfar:
        if int(r["min_warning_min"]) != 0:
            continue
        val = r["sensitivity_pct"].strip()
        if not val or val.lower() == "nan":
            continue
        curves[r["model"]].append((float(r["false_alerts_per_day"]), float(val)))
    for m in curves:
        curves[m].sort()

    tex = COMMON_PREAMBLE + r"""\usepackage[paperwidth=18.4cm,paperheight=8.6cm,margin=0.2cm]{geometry}
\begin{document}
\noindent\begin{tikzpicture}
\begin{axis}[
    name=axA,
    at={(0cm,0cm)},
    width=8.4cm, height=7.2cm,
    xmin=-1.18, xmax=1.18,
    ymin=0.1, ymax=12.75,
    xlabel={Paired diff.\ $\text{RMSE}_{\text{best}} - \text{RMSE}_{\text{HBML}}$ (mg/dL)},
    xlabel style={font=\scriptsize},
    ytick={1,2,3,4,5,6,7,8,9,10,11,12},
    yticklabels={""" + ",".join(reversed(row_labels)) + r"""},
    yticklabel style={font=\scriptsize},
    xtick={-1.0,-0.5,0,0.5,1.0},
    xticklabel style={font=\scriptsize},
    title={\textbf{(a)} External paired comparison (Hall 2018, $N=57$)},
    title style={font=\footnotesize, anchor=south, yshift=8pt},
    clip=false,
    axis lines=box,
]
\fill[hbmlgreen!12] (axis cs:-1.0,0.1) rectangle (axis cs:1.0,12.75);
\draw[hbmlgreen!70!black, densely dashed, line width=0.6pt] (axis cs:-1.0,0.1) -- (axis cs:-1.0,12.75);
\draw[hbmlgreen!70!black, densely dashed, line width=0.6pt] (axis cs:1.0,0.1) -- (axis cs:1.0,12.75);
\draw[black!55, dashed, line width=0.6pt] (axis cs:0.0,0.1) -- (axis cs:0.0,12.75);
\draw[black!20, line width=0.4pt] (axis cs:-1.18,4.5) -- (axis cs:1.18,4.5);
\draw[black!20, line width=0.4pt] (axis cs:-1.18,8.5) -- (axis cs:1.18,8.5);
""" + "\n".join(bars95) + "\n" + "\n".join(bars90) + r"""
\addplot[only marks, mark=*, mark size=2.1pt, color=hbmlblue, fill=hbmlblue] coordinates {""" + " ".join(sig_pts) + r"""};
\addplot[only marks, mark=*, mark size=2.1pt, draw=hbmlblue, fill=white, line width=0.8pt] coordinates {""" + " ".join(nonsig_pts) + r"""};
\node[anchor=south, font=\tiny, color=hbmlgreen!60!black] at (axis cs:0,12.80) {12/12 equivalent within $\pm 1.0$~mg/dL ($p_{\mathrm{TOST}} \le 3\times 10^{-4}$)};
\node[anchor=south east, font=\tiny, color=hbmlblue!80!black] at (axis cs:-0.05,0.15) {HBML trails $\leftarrow$};
\node[anchor=south west, font=\tiny, color=hbmlblue!80!black] at (axis cs:0.05,0.15) {$\rightarrow$ HBML better};
\end{axis}

\begin{axis}[
    name=axB,
    at={(axA.south east)}, xshift=1.35cm,
    width=7.4cm, height=7.2cm,
    xmin=0.22, xmax=1.04,
    ymin=0, ymax=84,
    xlabel={False alert clusters per scored participant-day},
    ylabel={Pre-onset sensitivity (\%)},
    xlabel style={font=\scriptsize},
    ylabel style={font=\scriptsize},
    xtick={0.25,0.45,0.65,0.85,1.0},
    xticklabel style={font=\scriptsize},
    ytick={0,20,40,60,80},
    yticklabel style={font=\scriptsize},
    grid=both, grid style={black!10},
    legend pos=north west,
    legend style={font=\tiny, draw=black!25, fill=white, fill opacity=0.92, text opacity=1, cells={anchor=west}},
    title={\textbf{(b)} Hall 30-min matched-FAR hypo sensitivity},
    title style={font=\footnotesize, anchor=south, yshift=6pt},
]
\addplot[hbmlred, mark=*, mark size=1.7pt, line width=1.4pt] coordinates {""" + " ".join(f"({x:.3f},{y:.2f})" for x, y in curves["HBML"]) + r"""};
\addlegendentry{HBML (\texttt{SFH-Sqrt})}
\addplot[hbmlgreen, dashed, mark=square*, mark size=1.6pt, line width=1.1pt] coordinates {""" + " ".join(f"({x:.3f},{y:.2f})" for x, y in curves["BestExpert"]) + r"""};
\addlegendentry{Best expert (ARIMA)}
\addplot[hbmlgrey, dotted, mark=triangle*, mark size=1.8pt, line width=1.1pt] coordinates {""" + " ".join(f"({x:.3f},{y:.2f})" for x, y in curves["Persistence"]) + r"""};
\addlegendentry{Persistence}
\node[anchor=south east, font=\tiny, fill=white, fill opacity=0.9, text opacity=1, inner sep=2pt, draw=black!20] at (axis cs:1.02,4)
  {\shortstack[r]{\textbf{At 1.0/day:} HBML $+11.9$~pp vs ARIMA\\(95\% CI $[+4.7,+19.2]$)}};
\end{axis}
\end{tikzpicture}
\end{document}
"""
    compile_tex_pdf(tex, os.path.join(OUT_DIR, "fig_external_validation.pdf"))


def build_mechanism_context_fig():
    pm_main = read_csv("overleaf/tables/pool_mechanism_summary.csv")
    pm_hall = read_csv("external_results/hall/pool_mechanism_summary.csv")
    all_pm = pm_main + pm_hall

    lu = {}
    for r in all_pm:
        hz = "0.5" if str(r["horizon"]) in ("0.5", "half") else str(int(float(r["horizon"])))
        lu[(r["dataset"], hz, r["context"], r["variant"])] = (
            float(r["mean_rmse"]),
            float(r["diff_vs_hbml"]),
            float(r["diff_ci95_low"]),
            float(r["diff_ci95_high"]),
        )

    rows_a = [
        ("weinstock", "0.5", "{Weinstock 0.5h}"),
        ("weinstock", "2", "{Weinstock 2h}"),
        ("weinstock", "5", "{Weinstock 5h}"),
        ("cgmacros", "0.5", "{CGMacros 0.5h}"),
        ("cgmacros", "2", "{CGMacros 2h}"),
        ("cgmacros", "5", "{CGMacros 5h}"),
        ("hall", "0.5", "{Hall 0.5h}"),
        ("hall", "2", "{Hall 2h}"),
        ("hall", "5", "{Hall 5h}"),
    ]
    pts = {"HBML-interleaved": [], "Uniform": [], "FixedConvex-CV": []}
    y_lbls = []
    n_r = len(rows_a)
    for idx, (ds, hz, lbl) in enumerate(rows_a):
        y = n_r - idx
        y_lbls.append(lbl)
        for v, dy in (("HBML-interleaved", 0.22), ("Uniform", 0.0), ("FixedConvex-CV", -0.22)):
            if (ds, hz, "full", v) not in lu:
                continue
            _, d, lo, hi = lu[(ds, hz, "full", v)]
            pts[v].append((d, y + dy, max(0.0, d - lo), max(0.0, hi - d)))

    def _ci_segments(lst, color):
        return "\n".join(
            rf"\draw[{color}, line width=0.9pt] (axis cs:{x - el:.4f},{y:.2f}) -- (axis cs:{x + eh:.4f},{y:.2f});"
            for x, y, el, eh in lst
        )

    ctxs = [("6", 1), ("12", 2), ("24", 3), ("full", 4)]
    def _curve_rel(ds, hz, variant):
        base = lu[(ds, hz, "full", "HBML")][0]
        return " ".join(f"({pos},{lu[(ds, hz, c, variant)][0] - base:+.3f})" for c, pos in ctxs)

    tex = COMMON_PREAMBLE + r"""\usepackage[paperwidth=18.4cm,paperheight=8.4cm,margin=0.2cm]{geometry}
\begin{document}
\noindent\begin{tikzpicture}
\begin{axis}[
    name=axA,
    at={(0cm,0cm)},
    width=8.4cm, height=7.1cm,
    xmin=-3.1, xmax=3.05,
    ymin=0.35, ymax=9.65,
    xlabel={RMSE diff.\ vs shared-state \texttt{HBML} (mg/dL, full)},
    xlabel style={font=\scriptsize},
    ytick={1,2,3,4,5,6,7,8,9},
    yticklabels={""" + ",".join(reversed(y_lbls)) + r"""},
    yticklabel style={font=\scriptsize},
    xtick={-3,-2,-1,0,1,2,3},
    xticklabel style={font=\scriptsize},
    legend pos=south east,
    legend style={font=\tiny, draw=black!25, fill=white, fill opacity=0.92, text opacity=1, cells={anchor=west}},
    title={\textbf{(a)} Delayed interleaving, Uniform, FixedConvex vs \texttt{HBML}},
    title style={font=\footnotesize, anchor=south, yshift=4pt},
]
\draw[black!55, dashed, line width=0.6pt] (axis cs:0,0.35) -- (axis cs:0,9.65);
\draw[black!20, line width=0.4pt] (axis cs:-3.1,3.5) -- (axis cs:3.05,3.5);
\draw[black!20, line width=0.4pt] (axis cs:-3.1,6.5) -- (axis cs:3.05,6.5);
""" + _ci_segments(pts["HBML-interleaved"], "hbmlblue") + "\n" + _ci_segments(pts["Uniform"], "hbmlorange") + "\n" + _ci_segments(pts["FixedConvex-CV"], "hbmlgrey") + r"""
\addplot[only marks, mark=*, mark size=1.8pt, color=hbmlblue] coordinates {""" + " ".join(f"({x:.3f},{y:.2f})" for x, y, _, _ in pts["HBML-interleaved"]) + r"""};
\addlegendentry{Interleaved ($h+1$)}
\addplot[only marks, mark=square*, mark size=1.7pt, color=hbmlorange] coordinates {""" + " ".join(f"({x:.3f},{y:.2f})" for x, y, _, _ in pts["Uniform"]) + r"""};
\addlegendentry{Uniform ($1/7$)}
\addplot[only marks, mark=triangle*, mark size=1.9pt, color=hbmlgrey] coordinates {""" + " ".join(f"({x:.3f},{y:.2f})" for x, y, _, _ in pts["FixedConvex-CV"]) + r"""};
\addlegendentry{FixedConvex-CV}
\node[anchor=north west, font=\tiny, color=black!65] at (axis cs:-2.98,9.55) {$\leftarrow$ lower than \texttt{HBML}};
\end{axis}

\begin{axis}[
    name=axB,
    at={(axA.south east)}, xshift=1.35cm,
    width=7.5cm, height=7.1cm,
    xmin=0.75, xmax=4.25,
    ymin=-0.55, ymax=8.7,
    xtick={1,2,3,4},
    xticklabels={6 hr, 12 hr, 24 hr, Full},
    xticklabel style={font=\scriptsize},
    ylabel={RMSE change vs full-history \texttt{HBML} (mg/dL, 2 hr)},
    ylabel style={font=\scriptsize},
    ytick={0,1,2,3,4,5,6},
    yticklabel style={font=\scriptsize},
    grid=both, grid style={black!10},
    legend pos=north east,
    legend style={font=\tiny, draw=black!25, fill=white, fill opacity=0.92, text opacity=1, cells={anchor=west}, legend columns=2},
    title={\textbf{(b)} Equal-context pool vs full pool (2-hr horizon)},
    title style={font=\footnotesize, anchor=south, yshift=4pt},
]
\draw[black!45, dashed, line width=0.5pt] (axis cs:0.75,0) -- (axis cs:4.25,0);
\addplot[hbmlblue, mark=*, mark size=1.8pt, line width=1.3pt] coordinates {""" + _curve_rel("weinstock", "2", "HBML-shortctx-pool") + r"""};
\addlegendentry{Weinstock short-ctx}
\addplot[hbmlblue, dashed, mark=o, mark size=1.7pt, line width=1.1pt] coordinates {""" + _curve_rel("weinstock", "2", "HBML") + r"""};
\addlegendentry{Weinstock full pool}
\addplot[hbmlred, mark=square*, mark size=1.7pt, line width=1.3pt] coordinates {""" + _curve_rel("cgmacros", "2", "HBML-shortctx-pool") + r"""};
\addlegendentry{CGMacros short-ctx}
\addplot[hbmlred, dashed, mark=square, mark size=1.6pt, line width=1.1pt] coordinates {""" + _curve_rel("cgmacros", "2", "HBML") + r"""};
\addlegendentry{CGMacros full pool}
\addplot[hbmlgreen, mark=triangle*, mark size=1.9pt, line width=1.2pt] coordinates {""" + _curve_rel("hall", "2", "HBML-shortctx-pool") + r"""};
\addlegendentry{Hall short-ctx}
\end{axis}
\end{tikzpicture}
\end{document}
"""
    compile_tex_pdf(tex, os.path.join(OUT_DIR, "fig_mechanism_context.pdf"))


def build_validation_robustness_fig():
    coh_rows = read_csv("validation_results/validation_cohort_sensitivity.csv")
    gap_rows = read_csv("validation_results/validation_gap_burden.csv")

    sets = [("retained_ge288", "Retained ($\\ge 288$, $N=195\\text{--}197$)", "hbmlblue", 0.17),
            ("all_scorable_ge1", "All scorable ($\\ge 1$, $N=200$)", "hbmlgreen", 0.0),
            ("common_all_conditions", "Common cohort ($N=195$)", "hbmlorange", -0.17)]
    hzs = [("0.5", 3, "{0.5 hr (ARIMA)}"), ("2", 2, "{2 hr (NODE)}"), ("5", 1, "{5 hr (Mean)}")]
    lu_c = {(r["dataset"], r["horizon"], r["context"], r["cohort_set"]): r for r in coh_rows}

    plot_a_cmds = []
    for key_set, lbl, col, dy in sets:
        pts_s = []
        for hz, y, _ in hzs:
            r = lu_c[("weinstock", hz, "full", key_set)]
            d = float(r["mean_diff_best_minus_hbml"])
            pts_s.append(f"({d:.4f},{y + dy:.2f})")
        plot_a_cmds.append(
            rf"\addplot[only marks, mark=*, mark size=2.1pt, color={col}] coordinates {{{' '.join(pts_s)}}};"
            + "\n" + rf"\addlegendentry{{{lbl}}}"
        )

    w_gap = {(r["horizon"], r["stratum"]): r for r in gap_rows if r["dataset"] == "weinstock"}
    strata = [("T1_LowGap", 1), ("T2_MedGap", 2), ("T3_HighGap", 3)]
    tick_lbls = []
    for st, _ in strata:
        r0 = w_gap[("0.5", st)]
        tick_lbls.append("{" + f"{st[:2]} ({float(r0['mean_missing_pct']):.1f}\\%)" + "}")

    c_05 = " ".join(f"({pos},{float(w_gap[('0.5', st)]['diff_best_minus_hbml']):.3f})" for st, pos in strata)
    c_2 = " ".join(f"({pos},{float(w_gap[('2', st)]['diff_best_minus_hbml']):.3f})" for st, pos in strata)
    c_hyp = " ".join(f"({pos},{float(w_gap[('0.5', st)]['hypo_sens_pct']):.1f})" for st, pos in strata)
    c_ppv = " ".join(f"({pos},{float(w_gap[('0.5', st)]['cluster_ppv_pct']):.1f})" for st, pos in strata)

    tex = COMMON_PREAMBLE + r"""\usepackage[paperwidth=18.4cm,paperheight=7.5cm,margin=0.2cm]{geometry}
\begin{document}
\noindent\begin{tikzpicture}
\begin{axis}[
    name=axA,
    at={(0cm,0cm)},
    width=8.1cm, height=6.2cm,
    xmin=-0.62, xmax=0.78,
    ymin=0.4, ymax=3.6,
    xlabel={Paired diff.\ $\text{RMSE}_{\text{best}} - \text{RMSE}_{\text{HBML}}$ (mg/dL)},
    xlabel style={font=\scriptsize},
    ytick={1,2,3},
    yticklabels={{5 hr (Mean)},{2 hr (NODE)},{0.5 hr (ARIMA)}},
    yticklabel style={font=\scriptsize},
    xtick={-0.5,-0.25,0,0.25,0.5,0.75},
    xticklabel style={font=\scriptsize},
    grid=both, grid style={black!10},
    legend pos=south east,
    legend style={font=\tiny, draw=black!25, fill=white, fill opacity=0.92, text opacity=1, cells={anchor=west}},
    title={\textbf{(a)} Weinstock cohort-retention sensitivity (full history)},
    title style={font=\footnotesize, anchor=south, yshift=4pt},
]
\fill[hbmlgreen!10] (axis cs:-0.62,0.4) rectangle (axis cs:0.78,3.6);
\draw[black!55, dashed, line width=0.6pt] (axis cs:0,0.4) -- (axis cs:0,3.6);
""" + "\n".join(plot_a_cmds) + r"""
\node[anchor=north west, font=\tiny, color=hbmlgreen!60!black, fill=white, fill opacity=0.9, text opacity=1, inner sep=1.5pt] at (axis cs:-0.59,3.52)
  {\shortstack[l]{All inside $\pm 1.0$~mg/dL\\($p_{\mathrm{TOST}} \le 0.002$)}};
\end{axis}

\begin{axis}[
    name=axB,
    at={(axA.south east)}, xshift=1.4cm,
    width=7.8cm, height=6.2cm,
    xmin=0.6, xmax=3.4,
    ymin=-0.05, ymax=1.10,
    xtick={1,2,3},
    xticklabels={""" + ",".join(tick_lbls) + r"""},
    xticklabel style={font=\scriptsize},
    xlabel={Weinstock missing-slot tertile (mean \% missing)},
    ylabel={Gain $\text{RMSE}_{\text{best}} - \text{RMSE}_{\text{HBML}}$ (mg/dL)},
    xlabel style={font=\scriptsize},
    ylabel style={font=\scriptsize},
    ytick={0,0.2,0.4,0.6,0.8,1.0},
    yticklabel style={font=\scriptsize},
    axis y line*=left,
    grid=both, grid style={black!10},
    legend pos=north west,
    legend style={font=\tiny, draw=black!25, fill=white, fill opacity=0.92, text opacity=1, cells={anchor=west}},
    title={\textbf{(b)} Missingness tertiles: RMSE gain \& 30-min safety},
    title style={font=\footnotesize, anchor=south, yshift=4pt},
]
\addplot[hbmlblue, mark=*, mark size=2.1pt, line width=1.4pt] coordinates {""" + c_05 + r"""};
\addlegendentry{0.5 hr RMSE gain}
\addplot[hbmlteal, mark=square*, mark size=2.0pt, line width=1.4pt] coordinates {""" + c_2 + r"""};
\addlegendentry{2 hr RMSE gain}
\end{axis}

\begin{axis}[
    at={(axB.south west)},
    width=7.8cm, height=6.2cm,
    xmin=0.6, xmax=3.4,
    ymin=35, ymax=88,
    axis y line*=right,
    axis x line=none,
    ylabel={30-min sensitivity / cluster PPV (\%)},
    ylabel style={font=\scriptsize, color=hbmlred},
    ytick={40,50,60,70,80},
    yticklabel style={font=\scriptsize, color=hbmlred},
    legend pos=south east,
    legend style={font=\tiny, draw=black!25, fill=white, fill opacity=0.92, text opacity=1, cells={anchor=west}},
]
\addplot[hbmlred, dashed, mark=triangle*, mark size=2.0pt, line width=1.1pt] coordinates {""" + c_ppv + r"""};
\addlegendentry{Cluster PPV (\%)}
\addplot[hbmlorange, dotted, mark=diamond*, mark size=2.0pt, line width=1.2pt] coordinates {""" + c_hyp + r"""};
\addlegendentry{Pre-onset sens.\ (\%)}
\end{axis}
\end{tikzpicture}
\end{document}
"""
    compile_tex_pdf(tex, os.path.join(OUT_DIR, "fig_validation_robustness.pdf"))


def main():
    build_external_validation_fig()
    build_mechanism_context_fig()
    build_validation_robustness_fig()


if __name__ == "__main__":
    main()
