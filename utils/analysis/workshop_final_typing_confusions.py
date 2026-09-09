#!/usr/bin/env python3
"""Build the final matched SAM-H tissue typing-confusion appendix reports.

The script reads canonical inference JSONs in a streaming fashion and writes
only derived paper-facing reports. It does not modify run artifacts.
"""

import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]
SOURCE_CSV = REPO / "reports" / "tissue_peft_vs_fullft_slideind.csv"
OUTPUT_MD = REPO / "reports" / "workshop_final_typing_confusions.md"
OUTPUT_TEX = REPO / "reports" / "workshop_final_typing_confusions.tex"

CLASSES = ["Immune", "Stromal", "Epithelial", "Melanocyte", "Other"]
SHORT = {"Immune": "I", "Stromal": "S", "Epithelial": "E", "Melanocyte": "M", "Other": "O"}
METHOD_SOURCES = {
    "PEFT": "peft_source_inference_json",
    "FullFT": "fullft_source_inference_json",
}


def stream_inference(path: Path):
    """Return foreground paired confusion, patch IDs and patch count."""
    confusion = [[0] * 6 for _ in range(6)]
    patch_ids = []
    in_images = False
    capturing = False
    bracket_depth = 0
    buffer = []
    matrices = 0

    with path.open(errors="strict") as handle:
        for line in handle:
            if line.startswith('  "image_metrics": {'):
                in_images = True
                continue
            if in_images and line.startswith('  "qc_filtered_metrics": {'):
                in_images = False
            if in_images:
                match = re.match(r'^    "(.+)": \{$', line.rstrip("\n"))
                if match:
                    patch_ids.append(match.group(1))

            if not capturing:
                if '"paired_confusion": [' in line:
                    capturing = True
                    fragment = line[line.index("[") :]
                    buffer = [fragment]
                    bracket_depth = fragment.count("[") - fragment.count("]")
            else:
                buffer.append(line)
                bracket_depth += line.count("[") - line.count("]")
                if bracket_depth == 0:
                    matrix = json.loads("".join(buffer).strip().rstrip(","))
                    if len(matrix) != 6 or any(len(row) != 6 for row in matrix):
                        raise RuntimeError(f"Unexpected paired-confusion shape in {path}")
                    for i in range(6):
                        for j in range(6):
                            confusion[i][j] += int(matrix[i][j])
                    matrices += 1
                    capturing = False
                    buffer = []

    if capturing:
        raise RuntimeError(f"Unterminated paired-confusion array in {path}")
    foreground = [row[1:] for row in confusion[1:]]
    return foreground, sorted(patch_ids), matrices


def patch_hash(ids):
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


def f1_by_class(matrix):
    values = []
    supports = []
    for i in range(5):
        true_support = sum(matrix[i])
        predicted = sum(matrix[j][i] for j in range(5))
        denominator = true_support + predicted
        values.append(2.0 * matrix[i][i] / denominator if denominator else None)
        supports.append(true_support)
    return values, supports


def dominant_pair(matrix):
    cells = [
        (matrix[i][j], CLASSES[i], CLASSES[j])
        for i in range(5)
        for j in range(5)
        if i != j
    ]
    maximum = max(value for value, _, _ in cells)
    winners = [(true, pred) for value, true, pred in cells if value == maximum]
    if len(winners) != 1:
        raise RuntimeError(f"Dominant-pair tie: {winners}, count={maximum}")
    return winners[0], maximum


def normalized(matrix):
    result = []
    for row in matrix:
        total = sum(row)
        result.append([value / total for value in row] if total else [None] * 5)
    return result


def mean_supported(matrices):
    result = []
    coverage = []
    for i in range(5):
        rows = [matrix[i] for matrix in matrices if matrix[i][0] is not None]
        coverage.append(len(rows))
        result.append([statistics.mean(row[j] for row in rows) for j in range(5)])
    return result, coverage


def fmt(value, digits=3):
    return "NA" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def signed(value, digits=3):
    return "NA" if value is None or not math.isfinite(value) else f"{value:+.{digits}f}"


def matrix_markdown(matrix, digits=3, normalized_values=False):
    lines = [
        "| True \\ Pred. | I | S | E | M | O |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in zip(CLASSES, matrix):
        if normalized_values:
            cells = [fmt(value, digits) for value in row]
        else:
            cells = [f"{int(value):,}" for value in row]
        lines.append(f"| {SHORT[name]} | " + " | ".join(cells) + " |")
    return lines


def build_records():
    with SOURCE_CSV.open(newline="") as handle:
        source_rows = [
            row for row in csv.DictReader(handle)
            if row["backbone"] == "CellViT-SAM-H"
        ]
    if len(source_rows) != 18:
        raise RuntimeError(f"Expected 18 SAM-H matched directions, found {len(source_rows)}")

    records = []
    for row in source_rows:
        record = {
            "tissue": row["tissue"],
            "fold": row["fold"],
            "patches": int(row["matched_test_patches"]),
            "expected_hash": row["test_patch_id_sha256"],
            "methods": {},
        }
        method_ids = {}
        for method, column in METHOD_SOURCES.items():
            source = REPO / row[column]
            inference_log = source.with_name("inference.log")
            config_path = source.with_name("config.yaml")
            if not source.is_file() or not inference_log.is_file() or not config_path.is_file():
                raise RuntimeError(f"Missing canonical provenance for {source}")
            if not re.search(r"checkpoint_10\.pth", inference_log.read_text(errors="replace")):
                raise RuntimeError(f"Inference did not explicitly use checkpoint_10: {source}")
            config = yaml.safe_load(config_path.read_text())
            if int(config.get("random_seed", -1)) != 42:
                raise RuntimeError(f"Unexpected seed in {config_path}")
            expected_taxonomy = {
                "Background": 0,
                "Immune": 1,
                "Stromal": 2,
                "Epithelial": 3,
                "Melanocyte": 4,
                "Other": 5,
            }
            if config.get("dataset_config", {}).get("nuclei_types") != expected_taxonomy:
                raise RuntimeError(f"Unexpected taxonomy in {config_path}")

            confusion, ids, matrix_count = stream_inference(source)
            if matrix_count != record["patches"] or len(ids) != record["patches"]:
                raise RuntimeError(
                    f"TEST patch-count mismatch in {source}: "
                    f"matrices={matrix_count}, ids={len(ids)}, expected={record['patches']}"
                )
            digest = patch_hash(ids)
            if digest != record["expected_hash"]:
                raise RuntimeError(f"TEST patch hash mismatch in {source}: {digest}")
            method_ids[method] = ids
            f1, support = f1_by_class(confusion)
            dominant, dominant_count = dominant_pair(confusion)
            record["methods"][method] = {
                "source": source.relative_to(REPO).as_posix(),
                "confusion": confusion,
                "normalized": normalized(confusion),
                "f1": f1,
                "support": support,
                "dominant": dominant,
                "dominant_count": dominant_count,
            }
        if method_ids["PEFT"] != method_ids["FullFT"]:
            raise RuntimeError(f"PEFT/FullFT TEST patch IDs differ for {row['tissue']} {row['fold']}")
        records.append(record)
    return records


def aggregate(records):
    result = {}
    for method in METHOD_SOURCES:
        mean_matrix, coverage = mean_supported(
            [record["methods"][method]["normalized"] for record in records]
        )
        mean_f1 = []
        f1_coverage = []
        for i in range(5):
            values = [
                record["methods"][method]["f1"][i]
                for record in records
                if record["methods"][method]["support"][i] > 0
                and record["methods"][method]["f1"][i] is not None
            ]
            mean_f1.append(statistics.mean(values))
            f1_coverage.append(len(values))
        counts = Counter(
            record["methods"][method]["dominant"] for record in records
        )
        result[method] = {
            "matrix": mean_matrix,
            "coverage": coverage,
            "mean_f1": mean_f1,
            "f1_coverage": f1_coverage,
            "dominant_counts": counts,
        }
    result["delta"] = [
        [
            result["PEFT"]["matrix"][i][j] - result["FullFT"]["matrix"][i][j]
            for j in range(5)
        ]
        for i in range(5)
    ]
    result["same_dominant"] = sum(
        record["methods"]["PEFT"]["dominant"]
        == record["methods"]["FullFT"]["dominant"]
        for record in records
    )
    return result


def write_markdown(records, agg):
    lines = [
        "# Final matched SAM-H tissue typing-confusion analysis",
        "",
        "**READ-ONLY DERIVED PAPER REPORT.** Canonical scientific artifacts were not modified.",
        "",
        "## Scope and method",
        "",
        "This audit uses exactly 18 matched tissue-specific reciprocal complete-slide TEST directions (nine tissues × Fold A/B), CellViT-SAM-H, seed 42. Every source was verified to contain the expected TEST patch count and patch-ID hash, the PEFT and FullFT patch-ID lists were identical within each direction, the saved config used seed 42 and the five-class STHELAR taxonomy, and `inference.log` explicitly loaded `checkpoint_10.pth`.",
        "",
        "Each raw matrix contains paired/matched nuclei only; unmatched detections are excluded. Rows are true classes and columns are predicted classes in the order I=Immune, S=Stromal, E=Epithelial, M=Melanocyte, O=Other. Each direction was row-normalized first and then directions were averaged equally, preventing large slides from dominating. A zero-support row is undefined and is omitted from that class-row average rather than treated as an all-zero error row. Thus I/S/E/O use n=18 directions and Melanocyte uses only the two Skin directions (n=2). Per-class F1 follows the same matched-nuclei and nonzero-true-support convention. Results are descriptive; no significance test is performed.",
        "",
        "Source matrix: `reports/tissue_peft_vs_fullft_slideind.csv`.",
        "",
        "## Macro-averaged row-normalized matrices",
        "",
        "### PEFT",
        "",
    ]
    lines.extend(matrix_markdown(agg["PEFT"]["matrix"], normalized_values=True))
    lines.extend(["", "Row coverage: " + ", ".join(
        f"{name} n={n}" for name, n in zip(CLASSES, agg["PEFT"]["coverage"])
    ) + ".", "", "### FullFT", ""])
    lines.extend(matrix_markdown(agg["FullFT"]["matrix"], normalized_values=True))
    lines.extend(["", "Row coverage: " + ", ".join(
        f"{name} n={n}" for name, n in zip(CLASSES, agg["FullFT"]["coverage"])
    ) + ".", "", "### PEFT minus FullFT", ""])
    delta_lines = [
        "| True \\ Pred. | I | S | E | M | O |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in zip(CLASSES, agg["delta"]):
        delta_lines.append(f"| {SHORT[name]} | " + " | ".join(signed(v) for v in row) + " |")
    lines.extend(delta_lines)

    lines.extend([
        "",
        "## Per-class matched-nuclei F1",
        "",
        "These are equal-direction means of the direction-specific class F1 values.",
        "",
        "| Class | PEFT | FullFT | PEFT−FullFT | Directions |",
        "|---|---:|---:|---:|---:|",
    ])
    for i, name in enumerate(CLASSES):
        p = agg["PEFT"]["mean_f1"][i]
        f = agg["FullFT"]["mean_f1"][i]
        n = min(agg["PEFT"]["f1_coverage"][i], agg["FullFT"]["f1_coverage"][i])
        lines.append(f"| {name} | {fmt(p)} | {fmt(f)} | {signed(p-f)} | {n} |")

    lines.extend([
        "",
        "## Dominant off-diagonal errors",
        "",
        "For comparability with the previously reported statement, the dominant error is the unique largest off-diagonal cell by raw matched-nucleus count within each direction. There were no ties. PEFT and FullFT share that pair in **{}/18 directions**; the exceptions are Skin A and Skin B.".format(agg["same_dominant"]),
        "",
        "| True→predicted pair | PEFT directions | FullFT directions |",
        "|---|---:|---:|",
    ])
    for true_name in CLASSES:
        for pred_name in CLASSES:
            if true_name == pred_name:
                continue
            pair = (true_name, pred_name)
            lines.append(
                f"| {true_name}→{pred_name} | "
                f"{agg['PEFT']['dominant_counts'].get(pair, 0)} | "
                f"{agg['FullFT']['dominant_counts'].get(pair, 0)} |"
            )

    lines.extend([
        "",
        "### Direction-level agreement",
        "",
        "| Tissue | Fold | PEFT dominant pair (count) | FullFT dominant pair (count) | Same |",
        "|---|:---:|---|---|:---:|",
    ])
    for record in records:
        p = record["methods"]["PEFT"]
        f = record["methods"]["FullFT"]
        p_pair = "→".join(p["dominant"])
        f_pair = "→".join(f["dominant"])
        same = "yes" if p["dominant"] == f["dominant"] else "no"
        lines.append(
            f"| {record['tissue']} | {record['fold']} | {p_pair} ({p['dominant_count']:,}) | "
            f"{f_pair} ({f['dominant_count']:,}) | {same} |"
        )

    lines.extend([
        "",
        "## Paper-ready wording",
        "",
        "### Main Discussion sentence",
        "",
        "Across the 18 matched complete-slide directions, PEFT and FullFT shared the largest matched-nucleus off-diagonal confusion in 16 cases, indicating that their typing errors largely followed the same slide-dependent class ambiguities even when class-wise F1 differed.",
        "",
        "### Appendix paragraph",
        "",
        "We row-normalized each direction before averaging, so every complete-slide direction contributed equally rather than in proportion to its matched-nucleus count. PEFT and FullFT had the same dominant off-diagonal error in 16 of 18 directions, with Stromal→Immune, Epithelial→Immune, Immune→Stromal, and Epithelial→Stromal recurring most often. Mean matched-nuclei F1 was similar for Epithelial (0.619 PEFT; 0.607 FullFT), while the largest difference occurred for Other (0.022; 0.120); Melanocyte remained weak for both methods and was evaluable only in the two Skin directions. These results show substantial overlap in the principal typing-error modes, while the class-level differences indicate that shared dominant confusions do not imply identical typing behavior.",
        "",
        "## Canonical source provenance",
        "",
        "| Tissue | Fold | TEST patches | Patch-ID SHA256 | PEFT source | FullFT source |",
        "|---|:---:|---:|---|---|---|",
    ])
    for record in records:
        lines.append(
            f"| {record['tissue']} | {record['fold']} | {record['patches']:,} | "
            f"`{record['expected_hash']}` | `{record['methods']['PEFT']['source']}` | "
            f"`{record['methods']['FullFT']['source']}` |"
        )

    lines.extend([
        "",
        "## Direction-level raw matched-nuclei matrices",
        "",
        "Rows are true labels; columns are predictions. These matrices provide the complete reconstruction requested for all 18 directions and both methods.",
        "",
    ])
    for record in records:
        lines.extend([f"### {record['tissue']} — Fold {record['fold']}", ""])
        for method in ("PEFT", "FullFT"):
            item = record["methods"][method]
            lines.extend([f"#### {method}", ""])
            lines.extend(matrix_markdown(item["confusion"]))
            lines.extend([
                "",
                f"Dominant off-diagonal: {item['dominant'][0]}→{item['dominant'][1]} ({item['dominant_count']:,}).",
                "",
            ])

    OUTPUT_MD.write_text("\n".join(lines).rstrip() + "\n")


def write_latex(agg):
    lines = [
        r"\begin{table}[t]",
        r"\caption{Matched-nucleus typing errors across 18 tissue-specific \SAMH{} complete-slide directions (seed 42). Class F1 is averaged equally over directions with nonzero matched true support; Melanocyte therefore uses only the two Skin directions. Dominant-pair counts indicate how often an ordered off-diagonal pair was the largest raw matched-nucleus error within a direction. PEFT and \FullFT{} shared the dominant pair in 16/18 directions.}",
        r"\label{tab:tissue-typing-confusions}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Class / dominant pair & PEFT & \FullFT{} & $\Delta$ (P$-$F) \\ ",
        r"\midrule",
        r"\multicolumn{4}{l}{\textit{Mean matched-nucleus class F1}} \\ ",
    ]
    for i, name in enumerate(CLASSES):
        p = agg["PEFT"]["mean_f1"][i]
        f = agg["FullFT"]["mean_f1"][i]
        suffix = r" $^{\dagger}$" if name == "Melanocyte" else ""
        lines.append(f"{name}{suffix} & {p:.3f} & {f:.3f} & {p-f:+.3f} \\\\ ")
    lines.extend([
        r"\midrule",
        r"\multicolumn{4}{l}{\textit{Directions where pair is the largest off-diagonal error}} \\ ",
    ])
    nonzero_pairs = set(agg["PEFT"]["dominant_counts"]) | set(agg["FullFT"]["dominant_counts"])
    ordered = sorted(
        nonzero_pairs,
        key=lambda pair: (
            -(agg["PEFT"]["dominant_counts"].get(pair, 0) + agg["FullFT"]["dominant_counts"].get(pair, 0)),
            CLASSES.index(pair[0]),
            CLASSES.index(pair[1]),
        ),
    )
    for pair in ordered:
        p = agg["PEFT"]["dominant_counts"].get(pair, 0)
        f = agg["FullFT"]["dominant_counts"].get(pair, 0)
        label = f"{pair[0]}$\\rightarrow${pair[1]}"
        lines.append(f"{label} & {p:d} & {f:d} & -- \\\\ ")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{1pt}",
        r"\parbox{0.96\linewidth}{\scriptsize $^{\dagger}$Melanocyte F1 uses $n=2$ supported directions; all other class-F1 rows use $n=18$. Rows are descriptive and are not treated as independent biological replicates.}",
        r"\end{table}",
    ])
    OUTPUT_TEX.write_text("\n".join(lines) + "\n")


def main():
    records = build_records()
    agg = aggregate(records)
    if agg["same_dominant"] != 16:
        raise RuntimeError(
            f"Previously reported dominant-pair agreement did not reproduce: {agg['same_dominant']}/18"
        )
    write_markdown(records, agg)
    write_latex(agg)
    print(f"Wrote {OUTPUT_MD.relative_to(REPO)}")
    print(f"Wrote {OUTPUT_TEX.relative_to(REPO)}")
    print(f"Dominant-pair agreement: {agg['same_dominant']}/18")


if __name__ == "__main__":
    main()
