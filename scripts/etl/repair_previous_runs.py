"""Revalida a precipitação de janeiro/2024 com evidência horária arquivada.

Consulta somente o mês afetado; preserva os outros meses e variáveis.
O cache permite reproduzir a correção sem rede. Não infere chuva a partir
da ausência de temperatura. Revisões da fonte ficam explícitas na auditoria.
"""
import json
import math
from pathlib import Path
import pandas as pd
from urllib.parse import urlencode
from urllib.request import urlopen
from extract_open_meteo_forecast_data import (
    BASE_URL, LAT, LON, MODEL, aggregate_to_daily, build_hourly_param,
)

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/outputs/datasets"
CACHE = ROOT / "data/outputs/open_meteo_raw/previous_runs_precip_2024-01.json"


def main():
    if CACHE.exists():
        payload = json.loads(CACHE.read_text(encoding="utf-8"))
    else:
        params = dict(latitude=LAT, longitude=LON, models=MODEL,
                      start_date="2024-01-01", end_date="2024-01-31",
                      timezone="America/Sao_Paulo",
                      hourly=build_hourly_param({"precipitation": "sum"}, [1, 2, 3]))
        with urlopen(BASE_URL + "?" + urlencode(params), timeout=60) as response:
            payload = json.load(response)
        if "error" in payload:
            raise ValueError(payload)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    hourly = pd.DataFrame(payload["hourly"]).rename(columns={"time": "datetime"})
    hourly["datetime"] = pd.to_datetime(hourly["datetime"])
    expected = pd.date_range("2024-01-01", "2024-02-01", freq="h", inclusive="left")
    if not pd.DatetimeIndex(hourly["datetime"]).equals(expected):
        raise ValueError("Resposta horária não cobre janeiro integralmente")
    columns = ["precipitation", *[f"precipitation_previous_day{n}" for n in (1, 2, 3)]]
    if not set(columns).issubset(hourly.columns):
        raise ValueError("Resposta sem todas as variáveis de precipitação")
    corrected = aggregate_to_daily(hourly, {"precipitation": "sum"}, [1, 2, 3]).set_index("date")
    original = pd.read_csv(BASE / "open_meteo_previous_runs_sp.csv", dtype=str).set_index("date")
    audit = []
    for column in columns:
        for day, value in corrected[column].items():
            key = day.strftime("%Y-%m-%d")
            old = float(original.at[key, column])
            if not ((pd.isna(old) and pd.isna(value)) or
                    math.isclose(old, value, rel_tol=1e-10, abs_tol=1e-10)):
                count = int(hourly.loc[hourly.datetime.dt.normalize().eq(day), column].count())
                audit.append(dict(date=day.strftime("%Y-%m-%d"), column=column,
                                  before=old, after=value, valid_hours=count))
                original.at[key, column] = str(value) if pd.notna(value) else float("nan")
    for lead in (1, 2, 3):
        # Só atualizar erros nas datas com precipitação alterada, preservando o restante.
        dates = {item["date"] for item in audit if item["column"] in
                 ("precipitation", f"precipitation_previous_day{lead}")}
        for day in dates:
            error = float(original.at[day, f"precipitation_previous_day{lead}"]) - float(original.at[day, "precipitation"])
            original.at[day, f"precipitation_error_day{lead}"] = str(error) if pd.notna(error) else float("nan")
            original.at[day, f"precipitation_abs_error_day{lead}"] = str(abs(error)) if pd.notna(error) else float("nan")
    report = BASE / "relatórios/open_meteo_correcao_janeiro.csv"
    report.parent.mkdir(parents=True, exist_ok=True)
    # Preservar auditoria inicial quando uma segunda execução não muda nada.
    if audit:
        pd.DataFrame(audit).to_csv(report, index=False)
    original.reset_index().to_csv(BASE / "open_meteo_previous_runs_sp.csv", index=False)
    print(f"Precipitação revalidada com dados horários: {len(audit)} células alteradas")


if __name__ == "__main__":
    main()
