#!/usr/bin/env python
import argparse
import json
import os
import re
from math import sqrt

import numpy as np
import pandas as pd


DISTANCE_MAP = {
    "wdistancec_nor": "WD",
    "edistancec_nor": "ED",
    "mdistancec_nor": "MMD",
}

PROPERTY_MAP = {
    "OO": "OO_dist",
    "OH": "OH_dist",
    "HOH": "HOH_dist",
    "ZOH": "ThetaOH_dist",
    "Hbond": "Hbonds",
    "OrderP": "OrderP",
}

# These 5 groups mirror visualiseEvaluateMeanInRadar.py intent.
GROUP_PATTERNS = {
    "Ref": r"^Ref(_C\d+)?$",
    "PPAFM2Exp_CoAll_L20_L1_Elatest_Only": r"^PPAFM2Exp_CoAll_L20_L1_Elatest_Only(_C\d+)?$",
    "PPAFM2Exp_CoAll_L20_L1_Elatest": r"^PPAFM2Exp_CoAll_L20_L1_Elatest(_C\d+)?$",
    "PPAFM2Exp_CoAll_L10_L10_Elatest_Only": r"^PPAFM2Exp_CoAll_L10_L10_Elatest_Only(_C\d+)?$",
    "PPAFM2Exp_CoAll_L10_L10_Elatest": r"^PPAFM2Exp_CoAll_L10_L10_Elatest(_C\d+)?$",
}

REFERENCE_PATTERN = {"Ref_Pure": r"^Ref_Pure(_C\d+)?$"}

CATEGORY_GROUPS = {
    "Pure": ["Ref_Pure"],
    "Handcrafted": ["Ref"],
    "Style translated (L20L1)": ["PPAFM2Exp_CoAll_L20_L1_Elatest_Only"],
    "Style translated (L10L10)": ["PPAFM2Exp_CoAll_L10_L10_Elatest_Only"],
    "Hybrid (L20L1)": ["PPAFM2Exp_CoAll_L20_L1_Elatest"],
    "Hybrid (L10L10)": ["PPAFM2Exp_CoAll_L10_L10_Elatest"],
}

CATEGORY_LATEX = {
    "Pure": r"$F_{\mathcal{U}}(\mathcal{V})$",
    "Handcrafted": r"$F_{\overline{\mathcal{V}}}(\mathcal{V})$",
    "Style translated (L20L1)": r"$F_{\widetilde{\mathcal{V}}}(\mathcal{V})$",
    "Style translated (L10L10)": r"$F_{\widetilde{\mathcal{V}}}(\mathcal{V})$",
    "Hybrid (L20L1)": r"$F_{\mathcal{V}^{\dagger}}(\mathcal{V})$",
    "Hybrid (L10L10)": r"$F_{\mathcal{V}^{\dagger}}(\mathcal{V})$",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export mean/error tables for WD/ED/MMD radar metrics."
    )
    parser.add_argument(
        "--input-json",
        default="../../processed_data/distribution_distances/similarities_Label_Top.json",
        help="Path to similarities JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default="../../results/meanInRadar/tables",
        help="Directory for table and diagnostics outputs.",
    )
    parser.add_argument(
        "--include-reference-pure",
        action="store_true",
        help="Also include Ref_Pure summary rows in the outputs.",
    )
    return parser.parse_args()


def build_distance_dataframe(similarities, distance_key):
    rows = []
    for structure, values in similarities.items():
        row = {"Structure": structure}
        for property_name, json_key in PROPERTY_MAP.items():
            row[property_name] = values[json_key][distance_key]
        rows.append(row)
    return pd.DataFrame(rows)


def calculate_stats(df, group_name, group_regex, distance_key):
    mask = df["Structure"].str.match(group_regex)
    selected = df.loc[mask].copy()

    records = []
    matched_structures = selected["Structure"].tolist()
    for property_name in PROPERTY_MAP:
        values = selected[property_name].dropna()
        n = int(values.count())
        mean = float(values.mean()) if n > 0 else np.nan
        std = float(values.std(ddof=1)) if n > 1 else np.nan
        sem = float(std / sqrt(n)) if n > 1 else np.nan

        records.append(
            {
                "group": group_name,
                "distance_key": distance_key,
                "metric": DISTANCE_MAP[distance_key],
                "property": property_name,
                "n": n,
                "mean": mean,
                "std": std,
                "sem": sem,
            }
        )

    return records, matched_structures


def save_markdown_table(df, output_path):
    md_df = df.copy()
    md_df["mean"] = md_df["mean"].map(lambda x: f"{x:.6f}" if pd.notna(x) else "nan")
    md_df["sem"] = md_df["sem"].map(lambda x: f"{x:.6f}" if pd.notna(x) else "nan")
    md_df["std"] = md_df["std"].map(lambda x: f"{x:.6f}" if pd.notna(x) else "nan")
    md_df["mean_pm_sem"] = md_df.apply(
        lambda row: f"{row['mean']} +/- {row['sem']}", axis=1
    )
    md_df = md_df[
        ["group", "metric", "property", "n", "mean_pm_sem", "std", "mean", "sem"]
    ]

    headers = list(md_df.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in md_df.iterrows():
        cells = [str(row[h]) for h in headers]
        lines.append("| " + " | ".join(cells) + " |")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def build_condensed_table(stats_df):
    rows = []
    for _, row in stats_df.iterrows():
        mean = row["mean"]
        sem = row["sem"]
        n = int(row["n"])
        if pd.notna(mean) and pd.notna(sem):
            value = f"{mean:.6f} +/- {sem:.6f}"
        elif pd.notna(mean):
            value = f"{mean:.6f}"
        else:
            value = "nan"
        rows.append(
            {
                "group": row["group"],
                "metric": row["metric"],
                "property": row["property"],
                "n": n,
                "mean_pm_sem": value,
            }
        )

    condensed = pd.DataFrame(rows)
    wide = condensed.pivot_table(
        index=["group", "metric"],
        columns="property",
        values="mean_pm_sem",
        aggfunc="first",
    ).reset_index()
    n_wide = condensed.pivot_table(
        index=["group", "metric"],
        columns="property",
        values="n",
        aggfunc="first",
    ).reset_index()

    n_columns = [col for col in n_wide.columns if col not in ["group", "metric"]]
    for col in n_columns:
        wide[f"n_{col}"] = n_wide[col]

    return wide


def _category_regex(group_names):
    regexes = []
    for name in group_names:
        if name in GROUP_PATTERNS:
            regexes.append(GROUP_PATTERNS[name])
        elif name in REFERENCE_PATTERN:
            regexes.append(REFERENCE_PATTERN[name])
        else:
            raise KeyError(f"Unknown group name in category mapping: {name}")
    return "(?:" + "|".join(regexes) + ")"


def _format_pm(mean, sem):
    if pd.notna(mean) and pd.notna(sem):
        return f"{mean:.6f} +/- {sem:.6f}"
    if pd.notna(mean):
        return f"{mean:.6f}"
    return "nan"


def build_category_tables(similarities, distance_key, score_label):
    df = build_distance_dataframe(similarities, distance_key)
    mins = df[list(PROPERTY_MAP.keys())].min()
    maxs = df[list(PROPERTY_MAP.keys())].max()

    display_map = {
        "OO": r"$d_{\mathrm{OO}}$",
        "OH": r"$d_{\mathrm{OH}}$",
        "HOH": r"$\theta_{\mathrm{HOH}}$",
        "ZOH": r"$\theta_{\mathrm{ZOH}}$",
        "Hbond": r"$(d_{\mathrm{O_d}\mathrm{O_a}}, \theta_{\mathrm{O_d}\mathrm{H}\mathrm{O_a}})$",
        "OrderP": r"$(S_k, S_g)$",
    }

    rows_normalized = []
    rows_detailed = []
    for category, groups in CATEGORY_GROUPS.items():
        regex = _category_regex(groups)
        subset = df[df["Structure"].str.match(regex)]
        row_normalized = {
            score_label: category,
            "latex category": CATEGORY_LATEX[category],
            "number of model replicas": int(subset["Structure"].nunique()),
        }
        row_detailed = {
            score_label: category,
            "latex category": CATEGORY_LATEX[category],
            "number of model replicas": int(subset["Structure"].nunique()),
        }
        for prop in PROPERTY_MAP.keys():
            values = subset[prop].dropna()
            n = int(values.count())
            mean = float(values.mean()) if n > 0 else np.nan
            std = float(values.std(ddof=1)) if n > 1 else np.nan
            sem = float(std / sqrt(n)) if n > 1 else np.nan

            denom = float(maxs[prop] - mins[prop])
            norm_mean = 1.0 - ((mean - float(mins[prop])) / denom) if n > 0 and denom != 0 else np.nan
            norm_sem = (sem / denom) if pd.notna(sem) and denom != 0 else np.nan

            raw_text = _format_pm(mean, sem)
            norm_text = _format_pm(norm_mean, norm_sem)
            row_normalized[display_map[prop]] = norm_text
            row_detailed[display_map[prop]] = f"({raw_text}), ({norm_text})"
        rows_normalized.append(row_normalized)
        rows_detailed.append(row_detailed)

    return pd.DataFrame(rows_normalized), pd.DataFrame(rows_detailed)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input_json, "r") as f:
        similarities = json.load(f)

    group_patterns = dict(GROUP_PATTERNS)
    if args.include_reference_pure:
        group_patterns.update(REFERENCE_PATTERN)

    all_records = []
    membership = {}

    for distance_key in DISTANCE_MAP:
        df = build_distance_dataframe(similarities, distance_key)
        for group_name, group_regex in group_patterns.items():
            records, matched_structures = calculate_stats(
                df=df,
                group_name=group_name,
                group_regex=group_regex,
                distance_key=distance_key,
            )
            all_records.extend(records)
            membership.setdefault(group_name, {
                "pattern": group_regex,
                "matched_structures": matched_structures,
                "n_structures": len(matched_structures),
            })

    stats_df = pd.DataFrame(all_records)
    stats_df.insert(0, "layer", "Top")

    long_csv = os.path.join(args.output_dir, "radar_stats_top_long.csv")
    stats_df.to_csv(long_csv, index=False)

    wide_df = stats_df.pivot_table(
        index=["group", "metric"],
        columns="property",
        values=["mean", "sem", "n"],
        aggfunc="first",
    )
    wide_df.columns = [f"{stat}_{prop}" for stat, prop in wide_df.columns]
    wide_df = wide_df.reset_index()

    wide_csv = os.path.join(args.output_dir, "radar_stats_top_wide.csv")
    wide_df.to_csv(wide_csv, index=False)

    markdown_out = os.path.join(args.output_dir, "radar_stats_top_long.md")
    save_markdown_table(stats_df, markdown_out)

    condensed_df = build_condensed_table(stats_df)
    condensed_csv = os.path.join(args.output_dir, "radar_stats_top_condensed.csv")
    condensed_df.to_csv(condensed_csv, index=False)

    category_table_specs = [
        ("wdistancec_nor", "WD score", "wd_category_table_top.csv", "wd_category_table_top_detailed.csv"),
        ("edistancec_nor", "ED score", "ed_category_table_top.csv", "ed_category_table_top_detailed.csv"),
        ("mdistancec_nor", "MMD score", "mmd_category_table_top.csv", "mmd_category_table_top_detailed.csv"),
    ]
    saved_category_tables = []
    for distance_key, score_label, normalized_name, detailed_name in category_table_specs:
        normalized_df, detailed_df = build_category_tables(
            similarities=similarities,
            distance_key=distance_key,
            score_label=score_label,
        )
        normalized_out = os.path.join(args.output_dir, normalized_name)
        detailed_out = os.path.join(args.output_dir, detailed_name)
        normalized_df.to_csv(normalized_out, index=False)
        detailed_df.to_csv(detailed_out, index=False)
        saved_category_tables.append(normalized_out)
        saved_category_tables.append(detailed_out)

    membership_out = os.path.join(args.output_dir, "group_membership_top.json")
    with open(membership_out, "w") as f:
        json.dump(membership, f, indent=2)

    print(f"Saved long table: {long_csv}")
    print(f"Saved wide table: {wide_csv}")
    print(f"Saved markdown table: {markdown_out}")
    print(f"Saved condensed table: {condensed_csv}")
    for category_table in saved_category_tables:
        print(f"Saved category table: {category_table}")
    print(f"Saved group membership: {membership_out}")


if __name__ == "__main__":
    main()
