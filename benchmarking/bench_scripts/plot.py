from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import defopt


def slugify(text):
    text = text.lower().replace(" ", "-")
    result = "".join(char for char in text if char.isalnum() or char == "-")
    while "--" in result:
        result = result.replace("--", "-")
    return result.strip("-")


def normalize_score_types(data):
    """
    Normalize score types across aligners for consistent plotting
    """
    data["score_size_str"] = data["score_size"].astype(str)

    mapping = {
        ("ish-aligner", "adaptive"): "adaptive",
        ("ish-aligner", "byte"): "byte",
        ("ish-aligner", "word"): "word",
        ("parasail_aligner", "8"): "byte",
        ("parasail_aligner", "16"): "word",
        ("parasail_aligner", "sat"): "adaptive",
    }

    def map_score_type(row):
        return mapping.get((row["aligner"], row["score_size_str"]))

    data["normalized_score_type"] = data.apply(map_score_type, axis=1)
    return data.dropna(subset=["normalized_score_type"])


def create_six_plots(data, architecture, output_dir):
    """
    Create six plots: full range and zoomed (0-300) for adaptive, byte, and word score types
    """
    score_types = ["adaptive", "byte", "word"]
    for score_type in score_types:
        create_plot(data, score_type, output_dir, zoom=False, architecture=architecture)
        create_plot(
            data,
            score_type,
            output_dir,
            zoom=True,
            max_query_len=300,
            architecture=architecture,
        )


def create_plot(
    data, score_type, output_dir: str, zoom=False, max_query_len=300, architecture=""
):
    filtered_data = data[data["normalized_score_type"] == score_type]
    if len(filtered_data) == 0:
        print(f"No data for score type: {score_type}")
        return

    plt.figure(figsize=(14, 9))
    aligners = filtered_data["aligner"].unique()

    simd_type = {
        "16/8": "SSE",
        "32/16": "AVX2",
        "64/32": "AVX512"
    }
    simd_widths = {
        "SSE": "16/8",
        "AVX2": "32/16",
        "AVX512": "64/32"
    }

    for aligner in sorted(aligners):
        aligner_data = filtered_data[filtered_data["aligner"] == aligner]
        width_configs = aligner_data.apply(
            lambda row: simd_type[f"{row['u8_width']}/{row['u16_width']}"], axis=1
        ).unique()

        for config in width_configs:
            u8_width, u16_width = map(int, simd_widths[config].split("/"))
            config_data = aligner_data[
                (aligner_data["u8_width"] == u8_width)
                & (aligner_data["u16_width"] == u16_width)
            ]

            if zoom:
                config_data = config_data[config_data["query_len"] <= max_query_len]

            config_data = config_data.sort_values("query_len")
            if len(config_data) == 0:
                continue

            marker = "o" if "ish" in aligner else "s"
            linestyle = "-" if "ish" in aligner else "--"

            if config == "SSE":
                color = "blue" if "ish" in aligner else "royalblue"
            elif config == "AVX2":
                color = "green" if "ish" in aligner else "limegreen"
            else:
                color = "red" if "ish" in aligner else "salmon"

            label = (
                f"{aligner.split('-')[0] if 'ish' in aligner else 'parasail'} {config}"
            )

            plt.plot(
                config_data["query_len"],
                config_data["gcups"],
                marker=marker,
                markersize=6,
                linestyle=linestyle,
                linewidth=2,
                color=color,
                label=label,
            )

    plt.xlabel("Query Length", fontsize=20)
    plt.ylabel("GCUPS (Giga Cell Updates Per Second)", fontsize=20)

    local_score_type_to_title = {
        "adaptive": "Adaptive",
        "word": "Unsigned 16-Bit ",
        "byte": "Unsigned 8-Bit"
    }
    semi_global_score_type_to_title = {
        "adaptive": "Adaptive",
        "word": "Signed 16-Bit ",
        "byte": "Signed 8-Bit"
    }
    score_type_modified = score_type
    if architecture.lower().startswith("local"):
        score_type_modified = local_score_type_to_title[score_type]
    elif architecture.lower().startswith("semi"):
        score_type_modified = semi_global_score_type_to_title[score_type]


    title_suffix = f" (Zoomed 0-{max_query_len})" if zoom else ""
    title = f"{architecture} GCUPS vs Query Length{title_suffix} - {score_type_modified} Scoring"
    plt.title(title, fontsize=22)

    y_max = 13

    if zoom:
        x_ticks = np.arange(0, max_query_len + 20, 20)
        plt.xlim(0, max_query_len)
        plt.ylim(0, y_max)
    else:
        max_len = filtered_data["query_len"].max()
        x_ticks = np.arange(0, max_len + 500, 500)
        plt.ylim(0, y_max)

    # Set y-axis ticks with increment of 1
    y_ticks = np.arange(0, y_max + 1, 1)
    
    plt.xticks(x_ticks, fontsize=16)  # Increase x-axis tick label font size
    plt.yticks(y_ticks, fontsize=16)  # Set y-axis ticks and increase font size
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(title="Aligner and SIMD Configuration", fontsize=20)
    plt.tight_layout()

    slug_title = slugify(title)
    filename = f"{output_dir}/plots/{slug_title}_{'zoomed' if zoom else 'full'}.png"
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Created {filename}")


def create_histogram_plot(data, output_dir, architecture=""):
    """
    Create histogram plots by binning query_len and averaging GCUPS per bin.
    Bins are defined for Protein-1 to Protein-4.
    """
    protein_bins = {
        "Protein-1 (0–200)": (0, 200),
        "Protein-2 (200–400)": (200, 400),
        "Protein-3 (400–600)": (400, 600),
        "Protein-4 (600-inf)": (600, np.inf),
    }

    name = {
        "1": f"1 {architecture}",
        "2": f"2 {architecture}s",
        "3": f"3 {architecture}s",
        "4": f"4 {architecture}s",
    }

    score_types = ["adaptive", "byte", "word"]
    for score_type in score_types:
        filtered_data = data[data["normalized_score_type"] == score_type]
        if len(filtered_data) == 0:
            print(f"No data for score type: {score_type}")
            continue

        plt.figure(figsize=(14, 9))
        aligners = filtered_data["aligner"].unique()
        x = np.arange(len(protein_bins))
        bar_width = 0.2
        shift = 0

        for aligner in sorted(aligners):
            aligner_data = filtered_data[filtered_data["aligner"] == aligner]
            devices = aligner_data.apply(
                lambda row: f"{row['devices']}", axis=1
            ).unique()

            for device in sorted(devices):
                config_data = aligner_data[aligner_data["devices"] == int(device)]

                bin_means = []
                for _, (low, high) in protein_bins.items():
                    bin_vals = config_data[
                        (config_data["query_len"] >= low)
                        & (config_data["query_len"] < high)
                    ]["gcups"]
                    bin_means.append(bin_vals.mean() if not bin_vals.empty else 0)

                color = (
                    "blue"
                    if device == "1"
                    else (
                        "green"
                        if device == "2"
                        else "yellow" if device == "3" else "red"
                    )
                )
                color = (
                    color
                    if "ish" in aligner
                    else {"blue": "royalblue", "green": "limegreen", "red": "salmon"}[
                        color
                    ]
                )

                label = f"{aligner.split('-')[0] if 'ish' in aligner else 'parasail'} {name[device]}"

                plt.bar(
                    x + shift,
                    bin_means,
                    width=bar_width,
                    label=label,
                    color=color,
                    edgecolor="black",
                )
                shift += bar_width

        plt.xlabel("Protein Bin", fontsize=20)
        plt.ylabel("Average GCUPS", fontsize=20)
        plt.title(
            f"{architecture} Average GCUPS by Protein Bin",
            fontsize=22,
        )
        plt.xticks(x + bar_width, list(protein_bins.keys()), fontsize=16)
        plt.yticks(fontsize=16)
        
        # Set y-axis to start at 0
        plt.ylim(bottom=0)
        
        plt.grid(True, linestyle="--", alpha=0.7, axis="y")
        plt.legend(title="Aligner", fontsize=18)
        plt.tight_layout()

        filename = (
            f"{output_dir}/plots/{slugify(architecture)}_{score_type}_protein_bins.png"
        )
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Created {filename}")


def generate_markdown_diff_table(data):
    """
    Create a markdown table comparing GCUPS % differences between aligners
    for each (query_len, normalized_score_type, width_config).
    """

    results = []
    grouped = data.groupby(
        ["query_len", "normalized_score_type", "u8_width", "u16_width"]
    )

    for (qlen, stype, u8, u16), group in grouped:
        aligner_data = group.groupby("aligner")["gcups"].mean()
        if "ish-aligner" in aligner_data and "parasail_aligner" in aligner_data:
            ish_gcups = aligner_data["ish-aligner"]
            parasail_gcups = aligner_data["parasail_aligner"]
            pct_diff = (
                100 * (ish_gcups - parasail_gcups) / parasail_gcups
                if parasail_gcups != 0
                else np.nan
            )
            results.append(
                {
                    "Query Length": qlen,
                    "Score Type": stype,
                    "Width Config": f"{u8}/{u16}",
                    "ISH GCUPS": f"{ish_gcups:.2f}",
                    "Parasail GCUPS": f"{parasail_gcups:.2f}",
                    "% Diff": f"{pct_diff:+.1f}%" if not np.isnan(pct_diff) else "N/A",
                }
            )

    df = pd.DataFrame(results)
    if df.empty:
        print("No comparable aligner configurations found.")
        return

    print("### GCUPS % Difference: ISH vs Parasail")
    print(df.to_markdown(index=False))


def generate_latex_binned_diff_table(data):
    """
    Create a LaTeX table comparing mean GCUPS % difference between aligners
    across binned query_len ranges.
    """

    bins = {
        "0–200": (0, 200),
        "200–400": (200, 400),
        "400–600": (400, 600),
        "600+": (600, np.inf),
    }

    score_type = "word"  # or loop over all types if needed
    filtered = data[data["normalized_score_type"] == score_type]

    rows = []
    for label, (low, high) in bins.items():
        bin_data = filtered[
            (filtered["query_len"] >= low) & (filtered["query_len"] < high)
        ]

        # Aggregate by width config
        for (u8, u16), group in bin_data.groupby(["u8_width", "u16_width"]):
            pivot = group.pivot_table(index="aligner", values="gcups", aggfunc="mean")

            if "ish-aligner" in pivot.index and "parasail_aligner" in pivot.index:
                ish_mean = pivot.loc["ish-aligner", "gcups"]
                parasail_mean = pivot.loc["parasail_aligner", "gcups"]
                pct_diff = (
                    100 * (ish_mean - parasail_mean) / parasail_mean
                    if parasail_mean != 0
                    else np.nan
                )

                rows.append(
                    {
                        "Bin": label,
                        "Width Config": f"{u8}/{u16}",
                        "ISH GCUPS": f"{ish_mean:.2f}",
                        "Parasail GCUPS": f"{parasail_mean:.2f}",
                        "% Diff": (
                            f"{pct_diff:.1f}\\%" if not np.isnan(pct_diff) else "N/A"
                        ),
                    }
                )

    # Output LaTeX
    print("\\begin{tabular}{l l r r r}")
    print("\\toprule")
    print("Bin & Width Config & ISH GCUPS & Parasail GCUPS & \\% Diff \\\\")
    print("\\midrule")
    for row in rows:
        print(
            f"{row['Bin']} & {row['Width Config']} & {row['ISH GCUPS']} & {row['Parasail GCUPS']} & {row['% Diff']} \\\\"
        )
    print("\\bottomrule")
    print("\\end{tabular}")


def main(
    benchmarks_csv: Path, architecture: str, output_dir: str, *, is_gpu: bool = False
):
    # Load and normalize data once
    data = pd.read_csv(benchmarks_csv)
    # architecture = "SG x64: c7a.xlarge - AMD EPYC 9R14"
    data = normalize_score_types(data)

    if is_gpu:
        # For GPU
        create_histogram_plot(data, output_dir, architecture)

    else:
        # Generate plots
        create_six_plots(data, architecture, output_dir)

    # For original tables in paper
    # generate_markdown_diff_table(data)
    # generate_latex_binned_diff_table(data)


if __name__ == "__main__":
    defopt.run(main)