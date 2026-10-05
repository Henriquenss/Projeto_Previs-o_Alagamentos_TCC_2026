"""Reconstrói a base integrada a partir dos CSVs locais, sem coleta de rede.

Uso: python scripts/etl/build_integrated_dataset.py
Os acumulados incluem o próprio dia. Esta base ainda não define um horizonte
operacional; selecione/desloque as variáveis antes de treinar previsões futuras.
"""
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[2] / "data/outputs/datasets"
CEMADEN_COLS = [
    "cemaden_precip_idw_mm", "cemaden_precip_mean_mm", "cemaden_precip_max_mm",
    "cemaden_precip_median_mm", "cemaden_intensity_max_hor",
    *[f"cemaden_precip_lag{n}d" for n in (1, 2, 3)],
    *[f"cemaden_precip_acc{n}d" for n in (3, 7, 14)],
]
METEO_COLS = ["precipitation_sum", "precipitation_sum_acc3d",
              "precipitation_sum_acc7d", "soil_moisture_0_to_7cm_mean"]


def build(base=BASE):
    raw = pd.read_csv(base / "final_data.csv")
    raw["date"] = pd.to_datetime(raw["date"], format="%d-%m-%Y", errors="raise")
    targets = raw.groupby("date").agg(
        has_flooding=("current_zone", lambda s: int(s.notna().any())),
        n_events=("current_zone", "count"),
        n_intransit=("status", lambda s: s.eq("Inativo Intransitável").sum()),
    ).reset_index()
    calendar = pd.DataFrame({"date": pd.date_range("2015-01-01", "2026-05-31")})
    result = calendar.merge(targets, on="date", how="left", validate="one_to_one")
    # Não transformar ausência de coleta em classe negativa.
    if result["has_flooding"].isna().any():
        raise ValueError("CGE não contém todos os dias; revisar coleta antes de integrar")
    for filename, columns in [("cemaden_sp_diario.csv", CEMADEN_COLS),
                               ("open_meteo_sp.csv", METEO_COLS)]:
        source = pd.read_csv(base / filename, parse_dates=["date"])
        result = result.merge(source[["date", *columns]], on="date", how="left",
                              validate="one_to_one")
    return result


if __name__ == "__main__":
    result = build()
    result.to_csv(BASE / "dataset_final_integrado.csv", index=False)
    print(f"Base integrada reconstruída: {len(result)} dias, {len(result.columns)} colunas")
