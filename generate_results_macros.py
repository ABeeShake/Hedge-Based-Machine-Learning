import csv
import os

def clean_name(name):
    return name.replace(" ", "").replace("(", "").replace(")", "").replace(".", "").replace("-", "")

def strip_latex(val):
    """Remove LaTeX formatting like \\textbf{...}* from a value string."""
    return val.replace("*", "").replace("\\textbf{", "").replace("}", "").strip()

def generate_macros():
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tables_dir = os.path.join(base_dir, "overleaf", "tables")
    output_file = os.path.join(base_dir, "overleaf", "sections", "generated_macros.tex")

    macros = []

    # ------------------------------------------------------------------ #
    # 1. RMSE per-model macros + percent-reduction macros
    # ------------------------------------------------------------------ #
    rmse_path = os.path.join(tables_dir, "rmse_pivot_mean.csv")
    context_map = [('6', 'SixHr'), ('12', 'TwelveHr'), ('24', 'TwentyFourHr'), ('full', 'Full')]

    if os.path.exists(rmse_path):
        # Load all rows into memory so we can do two passes
        all_rows = []
        with open(rmse_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append(row)

        # Helper to map horizon string to LaTeX macro fragment
        def horizon_label(h_raw):
            if h_raw == '0.5':
                return "ThirtyMin"
            elif h_raw == '2':
                return "TwoHour"
            elif h_raw == '5':
                return "FiveHour"
            return str(h_raw).replace(".", "")

        # --- Pass 1: per-model RMSE macros ---
        for row in all_rows:
            dataset  = clean_name(row['dataset']).capitalize()
            horizon  = horizon_label(row['horizon'])
            model    = clean_name(row['model'])
            for context_suffix, context_name in context_map:
                col_name = f"mean.{context_suffix}"
                if col_name in row and row[col_name]:
                    val = strip_latex(row[col_name])
                    macro_name = f"RMSEMean{dataset}{horizon}{model}{context_name}"
                    macros.append(f"\\newcommand{{\\{macro_name}}}{{{val}}}")

        # --- Pass 2: percent-reduction macros (HBML vs best expert) ---
        # Group rows by (dataset, horizon)
        from collections import defaultdict
        groups = defaultdict(list)
        for row in all_rows:
            key = (row['dataset'], row['horizon'])
            groups[key].append(row)

        macros.append("")
        macros.append("% RMSE percent-reduction macros: (best_expert - HBML) / best_expert * 100")
        for (dataset_raw, h_raw), rows in groups.items():
            dataset  = clean_name(dataset_raw).capitalize()
            horizon  = horizon_label(h_raw)

            for context_suffix, context_name in context_map:
                col_name = f"mean.{context_suffix}"
                hbml_val  = None
                best_val  = None  # lowest expert (non-HBML) value

                for row in rows:
                    raw = row.get(col_name, "")
                    if not raw:
                        continue
                    try:
                        v = float(strip_latex(raw))
                    except ValueError:
                        continue
                    if row['model'].startswith("HBML"):
                        hbml_val = v
                    else:
                        if best_val is None or v < best_val:
                            best_val = v

                if hbml_val is not None and best_val is not None and best_val > 0:
                    pct = (best_val - hbml_val) / best_val * 100
                    macro_name = f"RMSEReduction{dataset}{horizon}{context_name}"
                    macros.append(f"\\newcommand{{\\{macro_name}}}{{{pct:.1f}}}")

    # ------------------------------------------------------------------ #
    # 2. CEG macros
    # ------------------------------------------------------------------ #
    ceg_path = os.path.join(tables_dir, "max_ceg.csv")
    if os.path.exists(ceg_path):
        macros.append("")
        macros.append("% CEG zone percentage macros")

        def dataset_label(raw):
            """Map raw dataset name to a clean LaTeX macro fragment."""
            r = raw.strip().lower()
            if "weinstock" in r:
                return "Weinstock"
            if "cgmacros" in r or "cgm" in r:
                return "Cgmacros"
            # fallback: strip spaces/special chars
            return clean_name(raw).capitalize()

        with open(ceg_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Use the plain 'dataset' and 'horizon' columns — never dtab/htab
                # which contain raw LaTeX \multirow commands.
                dataset_raw = row.get('dataset', '').strip()
                h_raw       = row.get('horizon', '').strip()

                # Skip spanning rows where these columns are blank
                if not dataset_raw or not h_raw:
                    continue

                dataset = dataset_label(dataset_raw)

                if h_raw == '0.5':
                    horizon = "ThirtyMin"
                elif h_raw == '2':
                    horizon = "TwoHour"
                elif h_raw == '5':
                    horizon = "FiveHour"
                else:
                    horizon = h_raw.replace(".", "")

                context_raw = str(row.get('context', 'full')).strip()
                if context_raw == '6':
                    context_name = "SixHr"
                elif context_raw == '12':
                    context_name = "TwelveHr"
                elif context_raw == '24':
                    context_name = "TwentyFourHr"
                else:
                    context_name = "Full"

                # Individual zone macros
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
                        macro_name = f"CEG{dataset}{horizon}{context_name}{suffix}"
                        macros.append(f"\\newcommand{{\\{macro_name}}}{{{val_clean}}}")

                # Aggregate convenience macros used in results.tex
                # CD = C + D1 + D2 combined
                try:
                    cd = (zone_vals.get('C') or 0) + \
                         (zone_vals.get('DOne') or 0) + \
                         (zone_vals.get('DTwo') or 0)
                    macros.append(f"\\newcommand{{\\CEG{dataset}{horizon}{context_name}CD}}{{{cd:.2f}}}")
                except TypeError:
                    pass
                # E = E1 + E2 combined
                try:
                    e = (zone_vals.get('EOne') or 0) + (zone_vals.get('ETwo') or 0)
                    macros.append(f"\\newcommand{{\\CEG{dataset}{horizon}{context_name}E}}{{{e:.2f}}}")
                except TypeError:
                    pass

                # Context-less aliases for Full context (used as canonical in results.tex)
                if context_name == "Full":
                    all_suffixes = list(zone_map.values()) + ['CD', 'E']
                    for suffix in all_suffixes:
                        full_name  = f"CEG{dataset}{horizon}Full{suffix}"
                        alias_name = f"CEG{dataset}{horizon}{suffix}"
                        macros.append(f"\\newcommand{{\\{alias_name}}}{{\\{full_name}}}")

    # ------------------------------------------------------------------ #
    # Write output
    # ------------------------------------------------------------------ #
    with open(output_file, "w") as f:
        f.write("% Auto-generated macros from CSV data\n")
        f.write("\n".join(macros))

    print(f"Generated {len(macros)} macros in {output_file}")

if __name__ == "__main__":
    generate_macros()
