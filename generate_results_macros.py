import csv
import os

def clean_name(name):
    return name.replace(" ", "").replace("(", "").replace(")", "").replace(".", "").replace("-", "")

def strip_latex(val):
    """Remove LaTeX formatting like \\textbf{...}* from a value string."""
    return val.replace("*", "").replace("\\textbf{", "").replace("}", "").strip()

def horizon_label(h_raw):
    """Map raw horizon string/float to a clean LaTeX macro fragment."""
    h_str = str(h_raw).strip()
    if h_str in ['0.5', 'half', 'ThirtyMin']:
        return "ThirtyMin"
    elif h_str in ['2', '2.0', 'TwoHour']:
        return "TwoHour"
    elif h_str in ['5', '5.0', 'FiveHour']:
        return "FiveHour"
    return h_str.replace(".", "")

def generate_macros():
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tables_dir = os.path.join(base_dir, "overleaf", "tables")
    output_file = os.path.join(base_dir, "overleaf", "sections", "generated_macros.tex")

    macros = []
    context_map = [('6', 'SixHr'), ('12', 'TwelveHr'), ('24', 'TwentyFourHr'), ('full', 'Full')]

    # ------------------------------------------------------------------ #
    # 1. RMSE per-model macros (Mean and SD) + Percent-Reduction macros
    # ------------------------------------------------------------------ #
    rmse_path = os.path.join(tables_dir, "rmse_pivot_mean.csv")

    if os.path.exists(rmse_path):
        all_rows = []
        with open(rmse_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append(row)

        macros.append("% --- Per-Model RMSE Macros (Mean and SD) ---")
        # --- Pass 1: per-model RMSE Mean and SD macros ---
        for row in all_rows:
            dataset_raw = clean_name(row['dataset']).capitalize()
            dataset_variants = [dataset_raw]
            if dataset_raw.lower() == "cgmacros":
                dataset_variants = ["Cgmacros", "CGMacros"]

            horizon  = horizon_label(row['horizon'])
            model    = clean_name(row['model'])

            for context_suffix, context_name in context_map:
                col_name = f"mean.{context_suffix}"
                if col_name in row and row[col_name]:
                    val = strip_latex(row[col_name])
                    primary_name = f"RMSEMean{dataset_variants[0]}{horizon}{model}{context_name}"
                    macros.append(f"\\newcommand{{\\{primary_name}}}{{{val}}}")
                    for alt_ds in dataset_variants[1:]:
                        alt_name = f"RMSEMean{alt_ds}{horizon}{model}{context_name}"
                        macros.append(f"\\newcommand{{\\{alt_name}}}{{\\{primary_name}}}")

                sd_col_name = f"sd.{context_suffix}"
                if sd_col_name in row and row[sd_col_name]:
                    sd_val = strip_latex(row[sd_col_name])
                    primary_sd = f"RMSEsd{dataset_variants[0]}{horizon}{model}{context_name}"
                    macros.append(f"\\newcommand{{\\{primary_sd}}}{{{sd_val}}}")
                    for alt_ds in dataset_variants[1:]:
                        alt_sd = f"RMSEsd{alt_ds}{horizon}{model}{context_name}"
                        macros.append(f"\\newcommand{{\\{alt_sd}}}{{\\{primary_sd}}}")

        # --- Pass 2: percent-reduction macros (HBML vs best constituent expert) ---
        from collections import defaultdict
        groups = defaultdict(list)
        for row in all_rows:
            key = (row['dataset'], row['horizon'])
            groups[key].append(row)

        macros.append("")
        macros.append("% --- RMSE Percent-Reduction Macros: (best_expert - HBML) / best_expert * 100 ---")
        for (dataset_raw, h_raw), rows in groups.items():
            dataset_cap = clean_name(dataset_raw).capitalize()
            dataset_variants = [dataset_cap]
            if dataset_cap.lower() == "cgmacros":
                dataset_variants = ["Cgmacros", "CGMacros"]

            horizon  = horizon_label(h_raw)

            for context_suffix, context_name in context_map:
                col_name = f"mean.{context_suffix}"
                hbml_val = None
                best_val = None  # lowest expert (non-HBML) value

                for row in rows:
                    raw = row.get(col_name, "")
                    if not raw:
                        continue
                    try:
                        v = float(strip_latex(raw))
                    except ValueError:
                        continue
                    if row['model'].strip() == "HBML" or row['model'].startswith("HBML"):
                        hbml_val = v
                    else:
                        if best_val is None or v < best_val:
                            best_val = v

                if hbml_val is not None and best_val is not None and best_val > 0:
                    pct = (best_val - hbml_val) / best_val * 100
                    primary_red = f"RMSEReduction{dataset_variants[0]}{horizon}{context_name}"
                    macros.append(f"\\newcommand{{\\{primary_red}}}{{{pct:.1f}}}")
                    for alt_ds in dataset_variants[1:]:
                        alt_red = f"RMSEReduction{alt_ds}{horizon}{context_name}"
                        macros.append(f"\\newcommand{{\\{alt_red}}}{{\\{primary_red}}}")

    # ------------------------------------------------------------------ #
    # 2. CEG (Clarke Error Grid) macros
    # ------------------------------------------------------------------ #
    ceg_path = os.path.join(tables_dir, "max_ceg.csv")
    if os.path.exists(ceg_path):
        macros.append("")
        macros.append("% --- CEG Zone Percentage Macros ---")

        def dataset_label(raw):
            r = raw.strip().lower()
            if "weinstock" in r:
                return "Weinstock"
            if "cgmacros" in r or "cgm" in r:
                return "Cgmacros"
            return clean_name(raw).capitalize()

        with open(ceg_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                dataset_raw = row.get('dataset', '').strip()
                h_raw       = row.get('horizon', '').strip()

                if not dataset_raw or not h_raw:
                    continue

                dataset = dataset_label(dataset_raw)
                dataset_variants = [dataset]
                if dataset == "Cgmacros":
                    dataset_variants = ["Cgmacros", "CGMacros"]

                horizon = horizon_label(h_raw)

                context_raw = str(row.get('context', 'full')).strip()
                if context_raw == '6':
                    context_name = "SixHr"
                elif context_raw == '12':
                    context_name = "TwelveHr"
                elif context_raw == '24':
                    context_name = "TwentyFourHr"
                else:
                    context_name = "Full"

                zone_map = {
                    'A+Bpct': 'AB', 'Apct': 'A', 'Bpct': 'B',
                    'Cpct': 'C', 'D1pct': 'DOne', 'D2pct': 'DTwo',
                    'E1pct': 'EOne', 'E2pct': 'ETwo',
                }
                zone_vals = {}
                for col, suffix in zone_map.items():
                    val = row.get(col, "").strip()
                    if val:
                        val_clean = strip_latex(val)
                        try:
                            zone_vals[suffix] = float(val_clean)
                        except ValueError:
                            zone_vals[suffix] = None
                        primary_m = f"CEG{dataset_variants[0]}{horizon}{context_name}{suffix}"
                        macros.append(f"\\newcommand{{\\{primary_m}}}{{{val_clean}}}")
                        for alt_ds in dataset_variants[1:]:
                            macros.append(f"\\newcommand{{\\CEG{alt_ds}{horizon}{context_name}{suffix}}}{{\\{primary_m}}}")

                # Aggregate convenience macros
                # CD = C + D1 + D2 combined
                try:
                    cd = (zone_vals.get('C') or 0) + \
                         (zone_vals.get('DOne') or 0) + \
                         (zone_vals.get('DTwo') or 0)
                    cd_name = f"CEG{dataset_variants[0]}{horizon}{context_name}CD"
                    macros.append(f"\\newcommand{{\\{cd_name}}}{{{cd:.2f}}}")
                    for alt_ds in dataset_variants[1:]:
                        macros.append(f"\\newcommand{{\\CEG{alt_ds}{horizon}{context_name}CD}}{{\\{cd_name}}}")
                except TypeError:
                    pass

                # E = E1 + E2 combined
                try:
                    e = (zone_vals.get('EOne') or 0) + (zone_vals.get('ETwo') or 0)
                    e_name = f"CEG{dataset_variants[0]}{horizon}{context_name}E"
                    macros.append(f"\\newcommand{{\\{e_name}}}{{{e:.2f}}}")
                    for alt_ds in dataset_variants[1:]:
                        macros.append(f"\\newcommand{{\\CEG{alt_ds}{horizon}{context_name}E}}{{\\{e_name}}}")
                except TypeError:
                    pass

                # Context-less aliases for Full context (used as canonical in results.tex)
                if context_name == "Full":
                    all_suffixes = list(zone_map.values()) + ['CD', 'E']
                    for suffix in all_suffixes:
                        full_name  = f"CEG{dataset_variants[0]}{horizon}Full{suffix}"
                        alias_name = f"CEG{dataset_variants[0]}{horizon}{suffix}"
                        macros.append(f"\\newcommand{{\\{alias_name}}}{{\\{full_name}}}")
                        for alt_ds in dataset_variants[1:]:
                            macros.append(f"\\newcommand{{\\CEG{alt_ds}{horizon}{suffix}}}{{\\{full_name}}}")

    # ------------------------------------------------------------------ #
    # 3. Statistical Test Macros (Wilcoxon p, % improved, Paired Diff)
    # ------------------------------------------------------------------ #
    stats_path = os.path.join(tables_dir, "statistical_tests_best_expert.csv")
    if os.path.exists(stats_path):
        macros.append("")
        macros.append("% --- Paired Participant-Level Statistical Tests ---")
        with open(stats_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ds_raw = row['dataset'].strip().lower()
                ds_cap = "Weinstock" if "weinstock" in ds_raw else "Cgmacros"
                dataset_variants = [ds_cap]
                if ds_cap == "Cgmacros":
                    dataset_variants = ["Cgmacros", "CGMacros"]

                h_raw = row['horizon'].strip()
                h_name = horizon_label(h_raw)

                c_raw = row['context'].strip().lower()
                c_map = {'6': 'SixHr', '12': 'TwelveHr', '24': 'TwentyFourHr', 'full': 'Full'}
                ctx_name = c_map.get(c_raw, c_raw.capitalize())

                # Format p-value
                try:
                    p_val = float(row.get('wilcoxon_p', 1.0))
                    if p_val < 0.0001:
                        p_str = f"{p_val:.2e}"
                    else:
                        p_str = f"{p_val:.4f}"
                except ValueError:
                    p_str = row.get('wilcoxon_p', '')

                pct_imp = float(row.get('pct_improved', 0))
                n_imp = row.get('n_improved', '')
                n_total = row.get('n_participants', '')
                mean_diff = float(row.get('paired_diff_mean', 0))
                med_diff = float(row.get('paired_diff_median', 0))
                best_exp = row.get('expert', '')

                m_prim = dataset_variants[0]
                macros.append(f"\\newcommand{{\\WilcoxonP{m_prim}{h_name}{ctx_name}}}{{{p_str}}}")
                macros.append(f"\\newcommand{{\\PctImproved{m_prim}{h_name}{ctx_name}}}{{{pct_imp:.1f}}}")
                macros.append(f"\\newcommand{{\\NImproved{m_prim}{h_name}{ctx_name}}}{{{n_imp}}}")
                macros.append(f"\\newcommand{{\\NTotal{m_prim}{h_name}{ctx_name}}}{{{n_total}}}")
                macros.append(f"\\newcommand{{\\PairedDiffMean{m_prim}{h_name}{ctx_name}}}{{{mean_diff:.2f}}}")
                macros.append(f"\\newcommand{{\\PairedDiffMedian{m_prim}{h_name}{ctx_name}}}{{{med_diff:.2f}}}")
                macros.append(f"\\newcommand{{\\BestExpert{m_prim}{h_name}{ctx_name}}}{{{best_exp}}}")

                for alt_ds in dataset_variants[1:]:
                    macros.append(f"\\newcommand{{\\WilcoxonP{alt_ds}{h_name}{ctx_name}}}{{\\WilcoxonP{m_prim}{h_name}{ctx_name}}}")
                    macros.append(f"\\newcommand{{\\PctImproved{alt_ds}{h_name}{ctx_name}}}{{\\PctImproved{m_prim}{h_name}{ctx_name}}}")
                    macros.append(f"\\newcommand{{\\NImproved{alt_ds}{h_name}{ctx_name}}}{{\\NImproved{m_prim}{h_name}{ctx_name}}}")
                    macros.append(f"\\newcommand{{\\NTotal{alt_ds}{h_name}{ctx_name}}}{{\\NTotal{m_prim}{h_name}{ctx_name}}}")
                    macros.append(f"\\newcommand{{\\PairedDiffMean{alt_ds}{h_name}{ctx_name}}}{{\\PairedDiffMean{m_prim}{h_name}{ctx_name}}}")
                    macros.append(f"\\newcommand{{\\PairedDiffMedian{alt_ds}{h_name}{ctx_name}}}{{\\PairedDiffMedian{m_prim}{h_name}{ctx_name}}}")
                    macros.append(f"\\newcommand{{\\BestExpert{alt_ds}{h_name}{ctx_name}}}{{\\BestExpert{m_prim}{h_name}{ctx_name}}}")

    # ------------------------------------------------------------------ #
    # Write output
    # ------------------------------------------------------------------ #
    with open(output_file, "w") as f:
        f.write("% Auto-generated macros from CSV data\n")
        f.write("\n".join(macros) + "\n")

    print(f"Generated {len(macros)} macros in {output_file}")

if __name__ == "__main__":
    generate_macros()
