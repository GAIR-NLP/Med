#!/usr/bin/env python3
"""
Script to plot term1-4 absolute values.
"""
import argparse

from plot_paper_figures import plot_explain_figure

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot term1-4 absolute values")
    parser.add_argument(
        "experiment_names",
        type=str,
        nargs="+",
        help="List of experiment names to plot",
    )
    parser.add_argument(
        "--smoothing_factor",
        type=float,
        default=0.99,
        help="Smoothing factor (default: 0.99)",
    )
    parser.add_argument(
        "--smoothing_method",
        type=str,
        default="time_weighted_ema",
        choices=["none", "savgol", "ema", "time_weighted_ema"],
        help="Smoothing method (default: time_weighted_ema)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="paper_figures",
        help="Output directory (default: paper_figures)",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="term_absolute_values.png",
        help="Output filename (default: term_absolute_values.png)",
    )
    parser.add_argument(
        "--captions",
        type=str,
        nargs="*",
        default=None,
        help="Optional captions for each experiment (list of strings)",
    )
    parser.add_argument(
        "--aggregated_benchmarks",
        type=str,
        nargs="*",
        default=None,
        help="Benchmarks to aggregate (default: PERCEPTION_BENCHMARKS)",
    )

    args = parser.parse_args()

    plot_explain_figure(
        args.experiment_names,
        args.smoothing_factor,
        args.smoothing_method,
        args.output_dir,
        args.output_filename,
        args.captions,
        args.aggregated_benchmarks,
    )
