"""
Extração da Previsão Real D-1 — Open-Meteo Previous Runs API
TCC – Previsão de Alagamentos em São Paulo

CONTEXTO E MOTIVAÇÃO
─────────────────────────────────────────────────────────────────────────
Os notebooks anteriores (v1 e v2) usaram a Historical Weather API (ERA5),
que é REANÁLISE: uma reconstrução do que de fato aconteceu, calculada
DEPOIS do evento. Isso é adequado para nowcasting, mas é uma simplificação
otimista para forecasting — no mundo real, em D-1 você não tem acesso ao
ERA5 "verdadeiro" de D, você só tem a PREVISÃO que o modelo meteorológico
gerou em D-1 para D.

Este script usa a Previous Runs API, que arquiva exatamente isso: o valor
que cada variável tinha quando PREVISTA com X dias de antecedência, antes
do evento ocorrer. Isso é o que "o usuário via no D-1" de verdade.

Fonte : https://open-meteo.com/en/docs/previous-runs-api

CORREÇÃO IMPORTANTE (erro corrigido nesta versão)
─────────────────────────────────────────────────────────────────────────
Erro original ao tentar usar &daily=:
  {"error":true,"reason":"Data corrupted at path ''. Cannot initialize
   ForecastVariableDaily from invalid String value
   precipitation_sum,precipitation_sum_previous_day1."}

Causa: a Previous Runs API NÃO TEM parâmetro &daily=. A documentação só
lista "Hourly Weather Variables" — diferente da Historical Weather API
(ERA5), que aceita tanto &hourly= quanto &daily=. Os sufixos
"_previous_dayN" só existem no catálogo de variáveis horárias.

Correção: este script busca dados HORÁRIOS via &hourly= e agrega para
diário manualmente em Python (mesmo padrão usado no script do CEMADEN).

Limitação adicional: a maioria dos modelos só tem arquivo a partir de
Jan/2024, então este script cobre só uma janela recente do seu dataset
(2024-2026), não o histórico completo desde 2015.

O QUE O SCRIPT FAZ
─────────────────────────────────────────────────────────────────────────
Para cada hora, baixa via &hourly=:
  - precipitation                   → valor "mais recente" arquivado (referência)
  - precipitation_previous_day1     → previsão feita em D-1 para aquela hora
  - precipitation_previous_day2     → previsão feita em D-2
  - precipitation_previous_day3     → previsão feita em D-3

Depois agrega cada série horária para diário (soma para precipitação,
média/min/max para temperatura, máximo para vento), e calcula o erro de
previsão por horizonte (D-N previsto vs valor de referência).
"""

import requests
import pandas as pd
import numpy as np
import time

# ── Configurações ──────────────────────────────────────────────────────────

LAT, LON = -23.5505, -46.6333

# Previous Runs API tem cobertura completa só a partir de jan/2024 para a
# maioria dos modelos — valide com o teste de conectividade abaixo antes
# de baixar o período inteiro.
START_DATE = "2024-01-01"
END_DATE   = "2026-05-31"

OUTPUT_PATH = "open_meteo_previous_runs_sp.csv"

BASE_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

# Variáveis HORÁRIAS de interesse + tipo de agregação diária.
# IMPORTANTE: a API só aceita &hourly=, não existe &daily= nesta API.
HOURLY_VARS = {
    "precipitation"  : "sum",          # acumulado diário (mm)
    "temperature_2m" : "minmax_mean",  # gera _max, _min e _mean
    "wind_speed_10m" : "max",          # rajada/pico do dia
}
LEAD_DAYS = [1, 2, 3]  # gera _previous_day1, _previous_day2, _previous_day3

MODEL = "best_match"  # ou especifique um modelo fixo (ex: "ecmwf_ifs025") para
                       # garantir que o D-1 visto seja sempre do mesmo modelo


# ── Funções ────────────────────────────────────────────────────────────────

def build_hourly_param(vars_dict: dict, lead_days: list) -> str:
    """
    Monta a string de variáveis horárias incluindo o valor de referência
    (sem sufixo) e os offsets _previous_dayN para cada variável.
    """
    all_vars = list(vars_dict.keys())
    for var in vars_dict.keys():
        for lead in lead_days:
            all_vars.append(f"{var}_previous_day{lead}")
    return ",".join(all_vars)


def test_connectivity() -> bool:
    """
    Testa a API com uma janela pequena antes de baixar o período completo.
    Sempre rode isso primeiro — economiza tempo se algo estiver errado.
    """
    print("Testando conectividade e estrutura de resposta...")
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": "2024-06-01",
        "end_date": "2024-06-05",
        "hourly": build_hourly_param({"precipitation": "sum"}, [1]),
        "models": MODEL,
        "timezone": "America/Sao_Paulo",
    }
    try:
        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            print(f"  [erro da API] {data.get('reason')}")
            return False
        print("  ✓ Resposta OK. Colunas horárias retornadas:")
        for k in data.get("hourly", {}).keys():
            if k != "time":
                vals = data["hourly"][k]
                n_nan = sum(1 for v in vals if v is None)
                print(f"    - {k}  ({len(vals)} valores, {n_nan} nulos)")
        return True
    except Exception as e:
        print(f"  [erro] {e}")
        print("  Verifique se 'previous-runs-api.open-meteo.com' está liberado")
        print("  nas configurações de rede do seu ambiente.")
        return False


def fetch_previous_runs_hourly(start: str, end: str,
                                vars_dict: dict, lead_days: list,
                                model: str = MODEL) -> pd.DataFrame:
    """
    Busca os dados HORÁRIOS de previsão D-N para o período especificado.
    Faz por blocos de ~30 dias (dados horários geram muito mais pontos
    que diários, então o bloco é menor para evitar payloads excessivos).
    """
    hourly_param = build_hourly_param(vars_dict, lead_days)
    all_chunks = []

    start_dt = pd.to_datetime(start)
    end_dt   = pd.to_datetime(end)
    chunk_days = 30

    current = start_dt
    while current <= end_dt:
        chunk_end = min(current + pd.Timedelta(days=chunk_days), end_dt)

        params = {
            "latitude": LAT,
            "longitude": LON,
            "start_date": current.strftime("%Y-%m-%d"),
            "end_date": chunk_end.strftime("%Y-%m-%d"),
            "hourly": hourly_param,
            "models": model,
            "timezone": "America/Sao_Paulo",
        }

        print(f"  Buscando {current.date()} -> {chunk_end.date()}...")
        try:
            resp = requests.get(BASE_URL, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            if "error" in data:
                print(f"    [erro] {data.get('reason')}")
                current = chunk_end + pd.Timedelta(days=1)
                continue

            chunk_df = pd.DataFrame({"datetime": data["hourly"]["time"]})
            for var in vars_dict.keys():
                if var in data["hourly"]:
                    chunk_df[var] = data["hourly"][var]
                for lead in lead_days:
                    col = f"{var}_previous_day{lead}"
                    if col in data["hourly"]:
                        chunk_df[col] = data["hourly"][col]

            all_chunks.append(chunk_df)

        except Exception as e:
            print(f"    [erro] {e}")

        current = chunk_end + pd.Timedelta(days=1)
        time.sleep(0.5)

    if not all_chunks:
        return pd.DataFrame()

    df = pd.concat(all_chunks, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.drop_duplicates(subset="datetime").sort_values("datetime").reset_index(drop=True)
    return df


def aggregate_to_daily(df_hourly: pd.DataFrame,
                       vars_dict: dict, lead_days: list) -> pd.DataFrame:
    """
    Agrega as séries horárias (referência + cada lead) para nível diário.
    Precipitação usa soma; temperatura usa min/max/mean; vento usa max.
    """
    df_hourly = df_hourly.copy()
    df_hourly["date"] = df_hourly["datetime"].dt.date

    daily_frames = []

    for var, agg_type in vars_dict.items():
        cols_to_agg = [var] + [f"{var}_previous_day{lead}" for lead in lead_days]
        cols_to_agg = [c for c in cols_to_agg if c in df_hourly.columns]
        if not cols_to_agg:
            continue

        if agg_type == "sum":
            agg = df_hourly.groupby("date")[cols_to_agg].sum().reset_index()
        elif agg_type == "max":
            agg = df_hourly.groupby("date")[cols_to_agg].max().reset_index()
        elif agg_type == "minmax_mean":
            agg_mean = df_hourly.groupby("date")[cols_to_agg].mean().add_suffix("_mean")
            agg_max  = df_hourly.groupby("date")[cols_to_agg].max().add_suffix("_max")
            agg_min  = df_hourly.groupby("date")[cols_to_agg].min().add_suffix("_min")
            agg = pd.concat([agg_mean, agg_max, agg_min], axis=1).reset_index()
        else:
            agg = df_hourly.groupby("date")[cols_to_agg].mean().reset_index()

        daily_frames.append(agg)

    df_daily = daily_frames[0]
    for frame in daily_frames[1:]:
        df_daily = pd.merge(df_daily, frame, on="date", how="outer")

    df_daily["date"] = pd.to_datetime(df_daily["date"])
    return df_daily.sort_values("date").reset_index(drop=True)


# ── Execução principal ─────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Open-Meteo Previous Runs API - Previsao Real D-1/D-2/D-3")
    print("=" * 65)

    # 1. Teste de conectividade — SEMPRE rodar antes de baixar tudo
    ok = test_connectivity()
    if not ok:
        print("\n[!] Teste de conectividade falhou. Corrija antes de continuar.")
        print("    Se estiver rodando em ambiente sandboxed, libere o host:")
        print("    previous-runs-api.open-meteo.com")
        return

    print()

    # 2. Download do período completo (dados horários)
    print(f"[1/3] Baixando dados horários de {START_DATE} até {END_DATE}...")
    print(f"      Modelo: {MODEL}")
    print(f"      Variáveis base: {list(HOURLY_VARS.keys())}")
    print(f"      Horizontes: D-{LEAD_DAYS}")

    df_hourly = fetch_previous_runs_hourly(START_DATE, END_DATE, HOURLY_VARS, LEAD_DAYS)

    if df_hourly.empty:
        print("\n[ERRO] Nenhum dado retornado.")
        return

    print(f"\n  ✓ {len(df_hourly):,} leituras horárias obtidas.")

    # 3. Agregar para diário
    print("\n[2/3] Agregando para nível diário...")
    df_daily = aggregate_to_daily(df_hourly, HOURLY_VARS, LEAD_DAYS)
    print(f"  ✓ {len(df_daily)} dias obtidos.")

    # 4. Calcular erro de previsão por horizonte (foco em precipitação)
    print("\n[3/3] Calculando erro de previsão (D-N previsto vs referência)...")
    if "precipitation" in df_daily.columns:
        for lead in LEAD_DAYS:
            col = f"precipitation_previous_day{lead}"
            if col in df_daily.columns:
                df_daily[f"precipitation_error_day{lead}"] = df_daily[col] - df_daily["precipitation"]
                df_daily[f"precipitation_abs_error_day{lead}"] = df_daily[f"precipitation_error_day{lead}"].abs()

        print("\n  Erro absoluto médio de precipitação por horizonte:")
        for lead in LEAD_DAYS:
            col = f"precipitation_abs_error_day{lead}"
            if col in df_daily.columns:
                mae = df_daily[col].mean()
                print(f"    D-{lead}: MAE = {mae:.2f} mm")

    df_daily.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✓ Salvo em '{OUTPUT_PATH}'")
    print(f"  Shape: {df_daily.shape}")
    print(f"  Colunas: {df_daily.columns.tolist()}")
    print("=" * 65)


if __name__ == "__main__":
    main()
