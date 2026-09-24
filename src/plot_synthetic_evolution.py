import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_records(folder):
    jsonl_path = folder / "updates.jsonl"

    old_files = sorted(
        folder.glob("update_*_synthetic_changes.json")
    )
    light_files = sorted(
        folder.glob("update_*_summary.json")
    )

    formats_found = (
        int(jsonl_path.exists())
        + int(bool(old_files))
        + int(bool(light_files))
    )

    if formats_found > 1:
        raise ValueError(
            f"Mixed diagnostic formats in {folder}. "
            "Use a separate folder for each simulation."
        )

    if jsonl_path.exists():
        records = []

        with jsonl_path.open(
            "r",
            encoding="utf-8",
        ) as stream:
            for line_number, line in enumerate(
                stream,
                start=1,
            ):
                if not line.strip():
                    continue

                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {jsonl_path}, "
                        f"line {line_number}. "
                        "The run may still be writing "
                        "or may have been interrupted."
                    ) from exc
    else:
        files = light_files or old_files

        records = [
            json.loads(
                path.read_text(encoding="utf-8")
            )
            for path in files
        ]

    if not records:
        raise FileNotFoundError(
            f"No diagnostic records found in {folder}"
        )

    updates = [
        record["update"]
        for record in records
    ]

    if len(updates) != len(set(updates)):
        raise ValueError(
            f"Duplicate updates in {folder}. "
            "Do not save different runs "
            "in the same diagnostics folder."
        )

    return sorted(
        records,
        key=lambda record: record["update"],
    )


def build_table(records):
    rows = []

    for record in records:
        update = record["update"]
        local_n = record["local_samples"]
        local_overloaded = record["local_overloaded_count"]

        synthetic_n = 0
        synthetic_overloaded = 0

        def add_row(dataset, samples, overloaded):
            rows.append({
                "update": update,
                "plot_index": record["plot_index"],
                "dataset": dataset,
                "samples": samples,
                "overloaded_count": overloaded,
                "non_overloaded_count": samples - overloaded,
                "overloaded_pct": (
                    100 * overloaded / samples
                    if samples else np.nan
                ),
                "non_overloaded_pct": (
                    100 * (samples - overloaded) / samples
                    if samples else np.nan
                ),
            })

        add_row("Local training", local_n, local_overloaded)

        for item in record["changes"]:
            samples = item["samples_after"]
            overloaded = item["overloaded_count_after"]

            synthetic_n += samples
            synthetic_overloaded += overloaded

            add_row(
                f"Synthetic type {item['node_type']}",
                samples,
                overloaded,
            )

        add_row(
            "Synthetic only",
            synthetic_n,
            synthetic_overloaded,
        )
        add_row(
            "Local + synthetic",
            local_n + synthetic_n,
            local_overloaded + synthetic_overloaded,
        )

    return pd.DataFrame(rows)


def plot_evolution(table, output):
    datasets = list(table["dataset"].unique())
    updates = sorted(table["update"].unique())

    ncols = 2
    nrows = (len(datasets) + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(14, 3.5 * nrows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    for ax, dataset in zip(axes.flat, datasets):
        subset = (
            table[table["dataset"] == dataset]
            .set_index("update")
            .reindex(updates)
        )

        ax.plot(
            updates,
            subset["overloaded_pct"],
            label="Overloaded (1)",
            color="tab:orange",
            drawstyle="steps-post",
            marker=".",
            markersize=3,
        )

        ax.set_title(dataset)
        ax.set_xlabel("Gossip update (starting at 1)")
        ax.set_ylabel("Overloaded (%)")
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.25)

    for ax in list(axes.flat)[len(datasets):]:
        ax.set_visible(False)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=1,
        bbox_to_anchor=(0.5, 1.0),
    )

    fig.suptitle(
        "Overloaded percentage in the dataset used for each update",
        y=1.025,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(
        output / "class_proportions_over_updates.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    records = load_records(args.diagnostics_dir)
    table = build_table(records)

    table.to_csv(
        args.diagnostics_dir / "class_proportions_over_updates.csv",
        index=False,
    )
    plot_evolution(table, args.diagnostics_dir)

    print(f"Results saved in {args.diagnostics_dir.resolve()}")


if __name__ == "__main__":
    main()