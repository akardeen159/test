# Auto-converted from climate-disasters-pipeline.ipynb

"""
ENG 220 – Climate Change & Natural Disasters
Data Pipeline for Streamlit

This module provides functions to:
- Load and clean global temperature datasets
- Load and clean natural disaster datasets
- Aggregate to annual level
- Merge temperature + disaster counts
- Compute summary statistics and type frequencies

Files expected (in base_path or Kaggle input path):
- Gia_Bch_Nguyn_Earth_Temps_Cleaned.csv
- Berkeley_Earth_Temps_Cleaned.csv
- Josep_Ferrer_Temps_Cleaned.csv
- Baris_Dincer_Disasters_Cleaned.csv
- Shreyansh_Dangi_Disasters_Cleaned.csv
"""

from __future__ import annotations

import os
from typing import Tuple, Dict

import numpy as np
import pandas as pd

DATA_DIR_NAME = os.path.join("Cleaned Data", "Cleaned Data")


def _csv_path(base_path: str, filename: str) -> str:
    """Join base_path, data folder and filename into a full path."""
    return os.path.join(base_path, DATA_DIR_NAME, filename)



def load_temperature_data(
    base_path: str = ".",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and clean temperature datasets.

    Returns:
        temps_all:   long-format table with columns ['year', 'TempF', 'source']
        temps_annual: annual averages with columns ['year', 'TempF']
    """
    # Paths inside the "Cleaned Data" folder
    temps_gia_path   = _csv_path(base_path, "Gia_Bách_Nguyễn_Earth_Temps_Cleaned.csv")
    temps_berk_path  = _csv_path(base_path, "Berkeley_Earth_Temps_Cleaned.csv")
    temps_josep_path = _csv_path(base_path, "Josep_Ferrer_Temps_Cleaned.csv")

    # --- Gia: annual averages in Fahrenheit ---
    temps_gia = pd.read_csv(temps_gia_path)
    temps_gia = temps_gia.rename(
        columns={
            "Year": "year",
            "Average_Fahrenheit_Temperature": "TempF",
        }
    )
    temps_gia["source"] = "Gia_Bách_Nguyễn"

    # --- Berkeley Earth: monthly temps in °C -> °F ---
    temps_berk = pd.read_csv(temps_berk_path)
    temps_berk["date"] = pd.to_datetime(temps_berk["dt"], errors="coerce")
    temps_berk["year"] = temps_berk["date"].dt.year
    temps_berk["TempF"] = temps_berk["LandAndOceanAverageTemperature"] * 9 / 5 + 32
    temps_berk["source"] = "Berkeley_Earth"

    # --- Josep Ferrer: monthly temps in °F by country ---
    temps_josep = pd.read_csv(temps_josep_path)
    temps_josep["date"] = pd.to_datetime(temps_josep["EventDate"], errors="coerce")
    temps_josep["year"] = temps_josep["date"].dt.year
    temps_josep["TempF"] = pd.to_numeric(
        temps_josep["TemperatureFahrenheit"], errors="coerce"
    )
    temps_josep["source"] = "Josep_Ferrer"

    # Combine all temperature sources
    temps_all = pd.concat(
        [
            temps_gia[["year", "TempF", "source"]],
            temps_berk[["year", "TempF", "source"]],
            temps_josep[["year", "TempF", "source"]],
        ],
        ignore_index=True,
    )

    temps_all = temps_all.dropna(subset=["year", "TempF"])
    temps_all["year"] = temps_all["year"].astype(int)

    temps_annual = (
        temps_all.groupby("year", as_index=False)["TempF"]
        .mean()
        .sort_values("year")
    )

    return temps_all, temps_annual



def load_disaster_data(
    base_path: str = ".",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and clean disaster datasets.

    Returns:
        disasters_all: long-format events table with at least
                       ['event_date', 'year', 'disaster_type', 'source']
        disasters_per_year: aggregated table ['year', 'disaster_count']
    """
    # Paths inside the "Cleaned Data" folder
    dis_bar_path   = _csv_path(base_path, "Baris_Dincer_Disasters_Cleaned.csv")
    dis_shrey_path = _csv_path(base_path, "Shreyansh_Dangi_Disasters_Cleaned.csv")

    # --- Baris Dinçer disasters ---
    dis_bar = pd.read_csv(dis_bar_path)
    dis_bar["event_date"] = pd.to_datetime(dis_bar["EventDate"], errors="coerce")
    dis_bar["year"] = dis_bar["event_date"].dt.year

    dis_bar = dis_bar.rename(
        columns={
            "Var2": "region",
            "Var3": "disaster_group",
            "Var4": "disaster_subgroup",
            "Var5": "disaster_type",
        }
    )
    dis_bar["source"] = "Baris_Dincer"

    dis_bar_std = dis_bar[["event_date", "year", "disaster_type", "source"]]

    # --- Shreyansh Dangi disasters ---
    dis_shrey = pd.read_csv(dis_shrey_path)
    dis_shrey["event_date"] = pd.to_datetime(dis_shrey["Date"], errors="coerce")
    dis_shrey["year"] = dis_shrey["event_date"].dt.year
    dis_shrey["disaster_type"] = dis_shrey["DisasterType"]
    dis_shrey["source"] = "Shreyansh_Dangi"

    dis_shrey_std = dis_shrey[["event_date", "year", "disaster_type", "source"]]

    # Combine both disaster sources
    disasters_all = pd.concat([dis_bar_std, dis_shrey_std], ignore_index=True)

    disasters_all = disasters_all.dropna(subset=["year", "disaster_type"])
    disasters_all["year"] = disasters_all["year"].astype(int)

    disasters_per_year = (
        disasters_all.groupby("year", as_index=False)
        .size()
        .rename(columns={"size": "disaster_count"})
        .sort_values("year")
    )

    return disasters_all, disasters_per_year



# --------------------------------------------------------------------
# 2. Merged dataset + helpers
# --------------------------------------------------------------------

def build_merged_dataset(
    base_path: str = ".",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load everything and build the merged annual dataset.

    Returns:
        temps_annual:        ['year', 'TempF']
        disasters_per_year:  ['year', 'disaster_count']
        merged:              ['year', 'TempF', 'disaster_count']
    """
    _, temps_annual = load_temperature_data(base_path=base_path)
    _, disasters_per_year = load_disaster_data(base_path=base_path)

    merged = pd.merge(
        temps_annual, disasters_per_year, on="year", how="outer"
    ).sort_values("year")

    return temps_annual, disasters_per_year, merged


def compute_disaster_summary(merged: pd.DataFrame) -> Dict[str, float]:
    """
    Compute summary statistics for disaster_count (like Total Spent stats).

    Returns a dict with keys:
        Count, Mean, StdDev, Min, Median, Max, Sum
    """
    dc = merged["disaster_count"]

    summary_stats = {
        "Count": float(dc.count()),
        "Mean": float(dc.mean()),
        "StdDev": float(dc.std()),
        "Min": float(dc.min()),
        "Median": float(dc.median()),
        "Max": float(dc.max()),
        "Sum": float(dc.sum()),
    }
    return summary_stats


def disaster_type_counts(disasters_all: pd.DataFrame) -> pd.Series:
    """
    Return a Series with counts per disaster_type.
    """
    return disasters_all["disaster_type"].value_counts()


# Optional quick test if someone runs this module directly
if __name__ == "__main__":
    base = "."

    temps_all_df, temps_annual_df = load_temperature_data(base)
    disasters_all_df, disasters_per_year_df = load_disaster_data(base)
    temps_annual_df, disasters_per_year_df, merged_df = build_merged_dataset(base)
    stats = compute_disaster_summary(merged_df)
    type_counts = disaster_type_counts(disasters_all_df)

    print("Temperature annual head:")
    print(temps_annual_df.head(), "\n")

    print("Disasters per year head:")
    print(disasters_per_year_df.head(), "\n")

    print("Merged head:")
    print(merged_df.head(), "\n")

    print("Summary stats:", stats, "\n")
    print("Top disaster types:")
    print(type_counts.head())
