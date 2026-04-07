import os
import re

import numpy as np
import pandas as pd
from scipy import stats


def summarize_cycle_time_after_warmup(
    df: pd.DataFrame,
    warmup_time: float = 7400,
    end_time: float = 14800,
    exclude_seed: list | None = None,
):
    """
    Summarize cycle time statistics after the warm-up period using SEED-based pairing.

    This function:
    1. Identifies TIME and CYCLE columns by SEED
    2. Filters records within [warmup_time, end_time]
    3. Computes the mean cycle time for each seed
    4. Computes the overall mean, standard deviation, 95% confidence interval,
       and half-width across replication means

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame read from the "CycleTime" sheet in ALL_KPI_SUMMARY.xlsx
    warmup_time : float, default=7400
        End of the warm-up period
    end_time : float, default=14800
        End of the observation window
    exclude_seed : list | None, default=None
        List of seeds to exclude, e.g. [300]

    Returns
    -------
    replication_result_df : pd.DataFrame
        Per-seed summary statistics
    summary_dict : dict
        Overall statistics across per-seed mean cycle times
    """
    if exclude_seed is None:
        exclude_seed = []

    time_pattern = re.compile(r".*?_TIME\(SEED=(\d+)\)$")
    cycle_pattern = re.compile(r".*?_CYCLE\(SEED=(\d+)\)$")

    col_info = {}

    for col in df.columns:
        col_str = str(col).strip()

        m_time = time_pattern.fullmatch(col_str)
        if m_time:
            seed = int(m_time.group(1))
            if seed in col_info and "TIME" in col_info[seed]:
                raise ValueError(f"Duplicate TIME column detected for SEED={seed}.")
            col_info.setdefault(seed, {})["TIME"] = col
            continue

        m_cycle = cycle_pattern.fullmatch(col_str)
        if m_cycle:
            seed = int(m_cycle.group(1))
            if seed in col_info and "CYCLE" in col_info[seed]:
                raise ValueError(f"Duplicate CYCLE column detected for SEED={seed}.")
            col_info.setdefault(seed, {})["CYCLE"] = col
            continue

    valid_seeds = [
        seed
        for seed in sorted(col_info.keys())
        if "TIME" in col_info[seed]
        and "CYCLE" in col_info[seed]
        and seed not in exclude_seed
    ]

    if not valid_seeds:
        raise ValueError("No valid TIME/CYCLE column pairs were found by SEED.")

    replication_results = []

    for seed in valid_seeds:
        time_col = col_info[seed]["TIME"]
        cycle_col = col_info[seed]["CYCLE"]

        time_series = pd.to_numeric(df[time_col], errors="coerce")
        cycle_series = pd.to_numeric(df[cycle_col], errors="coerce")

        valid_mask = time_series.notna() & cycle_series.notna()
        time_series = time_series[valid_mask]
        cycle_series = cycle_series[valid_mask]

        window_mask = (time_series >= warmup_time) & (time_series <= end_time)
        selected_cycles = cycle_series[window_mask]

        n = len(selected_cycles)

        if n == 0:
            replication_mean = np.nan
            replication_std = np.nan
        elif n == 1:
            replication_mean = selected_cycles.mean()
            replication_std = np.nan
        else:
            replication_mean = selected_cycles.mean()
            replication_std = selected_cycles.std(ddof=1)

        replication_results.append(
            {
                "Seed": seed,
                "Count_in_Window": n,
                "CycleTime_Mean": replication_mean,
                "CycleTime_SD": replication_std,
            }
        )

    replication_result_df = pd.DataFrame(replication_results)

    valid_means = replication_result_df["CycleTime_Mean"].dropna()
    n_rep = len(valid_means)

    if n_rep == 0:
        raise ValueError("No valid cycle records were found within the selected time window.")

    overall_mean = valid_means.mean()

    if n_rep == 1:
        overall_std = np.nan
        ci_lower = np.nan
        ci_upper = np.nan
        half_width = np.nan
    else:
        overall_std = valid_means.std(ddof=1)
        t_critical = stats.t.ppf(0.975, df=n_rep - 1)
        half_width = t_critical * overall_std / np.sqrt(n_rep)
        ci_lower = overall_mean - half_width
        ci_upper = overall_mean + half_width

    summary_dict = {
        "warmup_time": warmup_time,
        "end_time": end_time,
        "num_replications_used": n_rep,
        "replication_mean_of_cycletime": overall_mean,
        "replication_sd_of_cycletime": overall_std,
        "ci95_lower": ci_lower,
        "ci95_upper": ci_upper,
        "ci95_half_width": half_width,
    }

    return replication_result_df, summary_dict


def summarize_replication_series(series: pd.Series):
    """
    Compute summary statistics for a replication-level KPI series.

    Parameters
    ----------
    series : pd.Series
        A numeric series containing one value per replication

    Returns
    -------
    dict
        Dictionary containing mean, standard deviation, 95% CI, and half-width
    """
    series = pd.to_numeric(series, errors="coerce").dropna()

    n = len(series)
    if n == 0:
        raise ValueError("No valid data were found for statistical analysis.")

    mean = series.mean()

    if n == 1:
        std = np.nan
        ci_lower = np.nan
        ci_upper = np.nan
        half_width = np.nan
    else:
        std = series.std(ddof=1)
        t_critical = stats.t.ppf(0.975, df=n - 1)
        half_width = t_critical * std / np.sqrt(n)
        ci_lower = mean - half_width
        ci_upper = mean + half_width

    return {
        "mean": mean,
        "std": std,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "half_width": half_width,
    }


def analyze_rejection_data(df: pd.DataFrame):
    """
    Analyze rejection rate data from the RejectionData sheet.
    Assumes:
    - Row 1: title
    - Row 2: headers
    - Row 3 onward: data
    """
    data = df.iloc[2:]
    rejection_series = data.iloc[:, 1]

    summary = summarize_replication_series(rejection_series)

    print("\n===== RejectionData =====")
    print(f"Mean = {summary['mean']:.6f}")
    print(f"SD   = {summary['std']:.6f}")
    print(f"95% CI = ({summary['ci_lower']:.6f}, {summary['ci_upper']:.6f})")
    print(f"95% CI Half-width = {summary['half_width']:.6f}")


def analyze_wait_data(df: pd.DataFrame):
    """
    Analyze average waiting time data from the WaitData sheet.
    Assumes:
    - Row 1: title
    - Row 2: headers
    - Row 3 onward: data
    """
    data = df.iloc[2:]

    aluminum_series = data.iloc[:, 1]
    harddisk_series = data.iloc[:, 2]

    summary_al = summarize_replication_series(aluminum_series)
    summary_hd = summarize_replication_series(harddisk_series)

    print("\n===== WaitData: AluminumBlock =====")
    print(f"Mean = {summary_al['mean']:.6f}")
    print(f"SD   = {summary_al['std']:.6f}")
    print(f"95% CI = ({summary_al['ci_lower']:.6f}, {summary_al['ci_upper']:.6f})")
    print(f"95% CI Half-width = {summary_al['half_width']:.6f}")

    print("\n===== WaitData: HardDisk =====")
    print(f"Mean = {summary_hd['mean']:.6f}")
    print(f"SD   = {summary_hd['std']:.6f}")
    print(f"95% CI = ({summary_hd['ci_lower']:.6f}, {summary_hd['ci_upper']:.6f})")
    print(f"95% CI Half-width = {summary_hd['half_width']:.6f}")


def analyze_inventory_data(df: pd.DataFrame):
    """
    Analyze average inventory or queue-related data from the InventoryData sheet.
    Assumes:
    - Row 1: title
    - Row 2: headers
    - Row 3 onward: data
    """
    data = df.iloc[2:]

    aluminum_series = data.iloc[:, 1]
    harddisk_series = data.iloc[:, 2]

    summary_al = summarize_replication_series(aluminum_series)
    summary_hd = summarize_replication_series(harddisk_series)

    print("\n===== InventoryData: AluminumBlock =====")
    print(f"Mean = {summary_al['mean']:.6f}")
    print(f"SD   = {summary_al['std']:.6f}")
    print(f"95% CI = ({summary_al['ci_lower']:.6f}, {summary_al['ci_upper']:.6f})")
    print(f"95% CI Half-width = {summary_al['half_width']:.6f}")

    print("\n===== InventoryData: HardDisk =====")
    print(f"Mean = {summary_hd['mean']:.6f}")
    print(f"SD   = {summary_hd['std']:.6f}")
    print(f"95% CI = ({summary_hd['ci_lower']:.6f}, {summary_hd['ci_upper']:.6f})")
    print(f"95% CI Half-width = {summary_hd['half_width']:.6f}")


def main():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    summary_file = os.path.join(dir_path, "ALL_KPI_SUMMARY.xlsx")

    # CycleTime analysis
    df_cycle = pd.read_excel(summary_file, sheet_name="CycleTime")
    rep_df, summary = summarize_cycle_time_after_warmup(df_cycle)

    print("\n===== CycleTime Summary After Warm-up =====")
    print(rep_df[["Seed", "Count_in_Window", "CycleTime_Mean"]])
    print(f"Mean = {summary['replication_mean_of_cycletime']:.6f}")
    print(f"SD   = {summary['replication_sd_of_cycletime']:.6f}")
    print(f"95% CI = ({summary['ci95_lower']:.6f}, {summary['ci95_upper']:.6f})")
    print(f"95% CI Half-width = {summary['ci95_half_width']:.6f}")

    rep_df.to_csv(
        os.path.join(dir_path, "CycleTime_Replication_Summary.csv"),
        index=False,
    )

    # Other KPI analyses
    df_rej = pd.read_excel(summary_file, sheet_name="RejectionData")
    df_wait = pd.read_excel(summary_file, sheet_name="WaitData")
    # df_inv = pd.read_excel(summary_file, sheet_name="InventoryData")

    analyze_rejection_data(df_rej)
    analyze_wait_data(df_wait)
    # Uncomment if needed
    # analyze_inventory_data(df_inv)


if __name__ == "__main__":
    main()