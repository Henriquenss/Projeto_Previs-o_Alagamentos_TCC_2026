"""
Consolidação e Validação dos Dados CEMADEN
TCC – Previsão de Alagamentos em São Paulo

Problema: os arquivos foram baixados manualmente com nomes inconsistentes:
  - 2015 : ABR-2015.csv, JAN-2015.csv ...  (nome = mês-ano)
  - 2016+: data (4).csv, data (5).csv ...   (nome = índice do download)

Este script:
  1. Lê TODOS os CSVs de QUALQUER subpasta de CEMADEN_ROOT
  2. Detecta o período de cada arquivo pela data interna (coluna datahora)
  3. Valida quais meses estão faltando no range 2015-01 → 2026-05
  4. Agrega para nível diário por estação e cria série SP (IDW)
  5. Salva relatório de gaps + dataset consolidado

Uso:
  python consolidate_cemaden.py

Saídas:
  cemaden_sp_diario.csv      → série diária consolidada para SP
  cemaden_estacoes.csv       → metadados das estações encontradas
  cemaden_relatorio_gaps.txt → meses faltando + resumo de qualidade
"""

import os
import re
import glob
import warnings
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
from datetime import date
from pathlib import Path
from dateutil.relativedelta import relativedelta

warnings.filterwarnings("ignore")

# ── Configurações ──────────────────────────────────────────────────────────

# Raiz onde estão as subpastas (2015/, 2016/, etc.) com os CSVs
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CEMADEN_ROOT = str(PROJECT_ROOT / "data/outputs/cemaden_raw")
OUTPUT_PATH = str(PROJECT_ROOT / "data/outputs/datasets")
EXCLUSIONS_PATH = Path(__file__).with_name("cemaden_exclusoes.csv")

START_PERIOD = (2015, 1)   # (ano, mês) início esperado
END_PERIOD   = (2026, 5)   # (ano, mês) fim esperado

OUTPUT_DAILY    = OUTPUT_PATH + "/cemaden_sp_diario.csv"
OUTPUT_STATIONS = OUTPUT_PATH + "/cemaden_estacoes.csv"
OUTPUT_REPORT   = OUTPUT_PATH + "/relatórios/cemaden_relatorio_gaps.txt"

SP_LAT, SP_LON = -23.5505, -46.6333
CEMADEN_ENCODING = "latin-1"

# Meses em PT para parsing de nomes como "ABR-2015"
MESES_PT = {
    "JAN": 1, "FEV": 2, "MAR": 3, "ABR": 4,
    "MAI": 5, "JUN": 6, "JUL": 7, "AGO": 8,
    "SET": 9, "OUT": 10, "NOV": 11, "DEZ": 12,
}


# ── Funções utilitárias ────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(a))


def all_months_in_range(start: tuple, end: tuple) -> list[tuple]:
    """Retorna lista de (ano, mês) de start até end inclusive."""
    months = []
    cur = date(start[0], start[1], 1)
    fin = date(end[0], end[1], 1)
    while cur <= fin:
        months.append((cur.year, cur.month))
        cur += relativedelta(months=1)
    return months


def guess_period_from_filename(filepath: str) -> tuple[int, int] | None:
    """
    Tenta inferir (ano, mês) pelo nome do arquivo.
    Reconhece padrões como:
      ABR-2015.csv  →  (2015, 4)
      JAN_2020.csv  →  (2020, 1)
      2016-03.csv   →  (2016, 3)
    Retorna None se não conseguir inferir.
    """
    name = os.path.splitext(os.path.basename(filepath))[0].upper()

    # Padrão: MES-ANO ou MES_ANO (ex: ABR-2015, JAN_2020)
    for sep in ["-", "_"]:
        parts = name.split(sep)
        if len(parts) >= 2:
            for i, part in enumerate(parts):
                if part in MESES_PT:
                    # procurar ano nos outros fragmentos
                    for other in parts:
                        if re.fullmatch(r"\d{4}", other):
                            return (int(other), MESES_PT[part])

    # Padrão: ANO-MES (ex: 2016-03)
    m = re.search(r"(\d{4})[-_](\d{1,2})", name)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        if 2010 <= y <= 2030 and 1 <= mo <= 12:
            return (y, mo)

    return None


def read_cemaden_file(filepath: str) -> pd.DataFrame | None:
    """
    Lê um CSV do CEMADEN e retorna DataFrame padronizado.
    Detecta automaticamente separador e encoding.
    """
    for enc in ["utf-8-sig", "utf-8", CEMADEN_ENCODING, "cp1252"]:
        for sep in [";", ","]:
            try:
                df = pd.read_csv(filepath, sep=sep, encoding=enc,
                                 low_memory=False, on_bad_lines="skip")
                if len(df.columns) >= 5:
                    break
            except Exception:
                continue
        else:
            continue
        break
    else:
        return None

    # Limpar BOM e espaços nos nomes de colunas
    df.columns = [re.sub(r"[ï»¿\ufeff]", "", c).strip() for c in df.columns]

    # Normalizar nomes de colunas (variações do CEMADEN)
    rename = {
        "municipio": "municipio", "Municipio": "municipio",
        "codEstacao": "cod_estacao", "CodEstacao": "cod_estacao", "Codigo": "cod_estacao",
        "uf": "uf", "UF": "uf",
        "nomeEstacao": "nome_estacao", "NomeEstacao": "nome_estacao", "Nome": "nome_estacao",
        "latitude": "lat", "Latitude": "lat",
        "longitude": "lon", "Longitude": "lon",
        "datahora": "datahora", "DataHora": "datahora", "dataHora": "datahora",
        "valorMedida": "valor_mm", "ValorMedido": "valor_mm", "Chuva": "valor_mm",
        "valormedida": "valor_mm",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    required = {"datahora", "valor_mm"}
    if not required.issubset(df.columns):
        return None

    # ── Converter datahora ────────────────────────────────────────────────
    # O CEMADEN usa dois formatos dependendo do período de download:
    #   Formato ISO (maioria): "2015-04-01 00:00:00.0"
    #   Formato BR  (alguns) : "01/02/2015 00:00:00"
    # ATENÇÃO: pd.to_datetime sem format pode confundir dd/mm com mm/dd.
    # Estratégia: detectar o formato pela primeira linha e aplicar consistentemente.
    raw_dt = df["datahora"].astype(str).str.strip()
    sample = raw_dt.dropna().iloc[0] if not raw_dt.dropna().empty else ""

    if re.match(r"\d{2}/\d{2}/\d{4}", sample):
        # Formato BR: dd/mm/yyyy HH:MM:SS
        parsed = pd.to_datetime(raw_dt, format="%d/%m/%Y %H:%M:%S", errors="coerce")
        # Fallback para variações sem segundos
        mask = parsed.isna()
        if mask.any():
            parsed[mask] = pd.to_datetime(raw_dt[mask], format="%d/%m/%Y %H:%M", errors="coerce")
    else:
        # Formato ISO: yyyy-mm-dd HH:MM:SS (com ou sem .0 no final)
        parsed = pd.to_datetime(raw_dt.str.replace(r"\.\d+$", "", regex=True), errors="coerce")

    df["datahora"] = parsed

    # ── Converter valor_mm (separador decimal pode ser vírgula) ───────────
    df["valor_mm"] = pd.to_numeric(
        df["valor_mm"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    df = df.dropna(subset=["datahora", "valor_mm"])
    df = df[df["valor_mm"] >= 0]

    # ── Converter lat/lon (idem: separador pode ser vírgula) ──────────────
    for col in ["lat", "lon"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )

    return df if not df.empty else None


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega leituras horárias/10min para nível diário por estação.
    A coluna datahora pode estar em UTC ou horário local — o CEMADEN
    declara UTC, mas alguns arquivos históricos já vêm em horário local.
    Este script trata como horário de SP (UTC-3) para consistência.
    """
    # Se timezone-naive, assumir America/Sao_Paulo
    if df["datahora"].dt.tz is None:
        df["datahora"] = df["datahora"].dt.tz_localize("America/Sao_Paulo",
                                                        ambiguous="NaT",
                                                        nonexistent="NaT")
    else:
        df["datahora"] = df["datahora"].dt.tz_convert("America/Sao_Paulo")

    df["date"] = df["datahora"].dt.date

    group_cols = ["date"]
    if "cod_estacao" in df.columns:
        group_cols = ["cod_estacao"] + group_cols

    agg = df.groupby(group_cols, dropna=False).agg(
        precip_total_mm   = ("valor_mm", "sum"),
        precip_max_hor_mm = ("valor_mm", "max"),   # legado: máximo por leitura, NÃO por hora
        n_leituras        = ("valor_mm", "count"),
    ).reset_index()

    agg["date"] = pd.to_datetime(agg["date"])

    # Preservar metadados de estação
    if "cod_estacao" in df.columns:
        meta_cols = [c for c in ["cod_estacao", "nome_estacao", "lat", "lon"] if c in df.columns]
        meta = df.groupby("cod_estacao")[meta_cols[1:]].first().reset_index()
        agg  = pd.merge(agg, meta, on="cod_estacao", how="left")

    return agg


def exclude_station_days(df_daily, exclusions):
    """Retira estação-dia inteira; preserva outras estações e não imputa zero."""
    marked = df_daily.merge(
        exclusions, on=["cod_estacao", "date"], how="left", validate="many_to_one"
    )
    excluded = marked[marked["motivo"].notna()].copy()
    kept = marked[marked["motivo"].isna()][df_daily.columns].copy()
    return kept, excluded


def build_sp_series(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Consolida dados diários por estação em série única para SP via IDW.
    """
    has_coords = "lat" in df_daily.columns and df_daily["lat"].notna().any()

    if has_coords and "cod_estacao" in df_daily.columns:
        meta = (df_daily.groupby("cod_estacao")[["lat", "lon"]]
                .first().reset_index().dropna())
        meta["dist"] = meta.apply(
            lambda r: haversine(r["lat"], r["lon"], SP_LAT, SP_LON), axis=1
        )
        meta["dist"]   = meta["dist"].clip(lower=0.1)
        meta["weight"] = 1.0 / meta["dist"]
        meta["weight"] /= meta["weight"].sum()

        df_w = pd.merge(df_daily, meta[["cod_estacao", "weight"]],
                        on="cod_estacao", how="left")

        def _idw(group):
            w = group["weight"].fillna(0).values
            v = group["precip_total_mm"].values
            mask = ~np.isnan(v) & (w > 0)
            if not mask.any():
                return np.nan
            return np.average(v[mask], weights=w[mask] / w[mask].sum())

        idw = (df_w.groupby("date")
               .apply(_idw)
               .reset_index(name="cemaden_precip_idw_mm"))

        rest = df_daily.groupby("date").agg(
            cemaden_precip_mean_mm    = ("precip_total_mm",   "mean"),
            cemaden_precip_max_mm     = ("precip_total_mm",   "max"),
            cemaden_precip_median_mm  = ("precip_total_mm",   "median"),
            cemaden_intensity_max_hor = ("precip_max_hor_mm", "max"),
            cemaden_n_estacoes        = ("precip_total_mm",   "count"),
        ).reset_index()

        return pd.merge(idw, rest, on="date", how="outer")

    return df_daily.groupby("date").agg(
        cemaden_precip_mean_mm    = ("precip_total_mm",   "mean"),
        cemaden_precip_max_mm     = ("precip_total_mm",   "max"),
        cemaden_precip_median_mm  = ("precip_total_mm",   "median"),
        cemaden_intensity_max_hor = ("precip_max_hor_mm", "max"),
        cemaden_n_estacoes        = ("precip_total_mm",   "count"),
    ).reset_index()


# ── Pipeline principal ─────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("CEMADEN — Consolidação e Validação de Dados")
    print("=" * 65)

    Path(OUTPUT_REPORT).parent.mkdir(parents=True, exist_ok=True)
    exclusions = pd.read_csv(EXCLUSIONS_PATH, parse_dates=["date"])
    # 1. Descobrir todos os CSVs recursivamente
    pattern = os.path.join(CEMADEN_ROOT, "**", "*.csv")
    files   = sorted(glob.glob(pattern, recursive=True))
    print(f"\n[1/5] Arquivos .csv encontrados: {len(files)}")

    if not files:
        print(f"\n[!] Nenhum arquivo encontrado em '{CEMADEN_ROOT}'")
        print(f"    Verifique se o caminho está correto.")
        return

    # 2. Ler e catalogar cada arquivo
    print("\n[2/5] Lendo arquivos e inferindo períodos...")

    catalog   = []   # [{file, year, month, n_rows, period_source}]
    all_daily = []
    errors    = []

    for filepath in files:
        fname = os.path.basename(filepath)

        df = read_cemaden_file(filepath)
        if df is None:
            errors.append((fname, "Falha na leitura / colunas não reconhecidas"))
            continue

        # Período pela data interna (mais confiável)
        period_min = df["datahora"].min()
        period_max = df["datahora"].max()
        year_int  = period_min.year  if pd.notna(period_min) else None
        month_int = period_min.month if pd.notna(period_min) else None
        source    = "data interna"

        # Fallback: tentar pelo nome do arquivo
        if year_int is None:
            guessed = guess_period_from_filename(filepath)
            if guessed:
                year_int, month_int = guessed
                source = "nome do arquivo"

        catalog.append({
            "arquivo"       : fname,
            "caminho"       : filepath,
            "ano"           : year_int,
            "mes"           : month_int,
            "linhas"        : len(df),
            "n_estacoes"    : df["cod_estacao"].nunique() if "cod_estacao" in df.columns else "?",
            "data_min"      : str(period_min)[:10] if pd.notna(period_min) else "?",
            "data_max"      : str(period_max)[:10] if pd.notna(period_max) else "?",
            "periodo_fonte" : source,
        })

        # Agregar para diário
        daily = aggregate_daily(df)
        all_daily.append(daily)

        status = f"({year_int}-{month_int:02d})" if year_int else "(?)"
        print(f"  OK  {fname:30s} {status}  {len(df):>8,} leituras  {daily['date'].nunique()} dias")

    print(f"\n  Lidos: {len(catalog)} OK  |  Erros: {len(errors)}")
    if errors:
        print("  Arquivos com erro:")
        for fname, msg in errors:
            print(f"    ✗ {fname}: {msg}")
        raise RuntimeError("Falha em arquivos brutos; saídas não foram substituídas.")

    # 3. Validar gaps
    print("\n[3/5] Validando cobertura temporal...")

    expected   = all_months_in_range(START_PERIOD, END_PERIOD)
    found      = set((r["ano"], r["mes"]) for r in catalog if r["ano"] is not None)
    missing    = sorted(set(expected) - found)

    cat_df = pd.DataFrame(catalog).sort_values(["ano", "mes"])

    print(f"  Meses esperados : {len(expected)}")
    print(f"  Meses encontrados: {len(found)}")
    print(f"  Meses faltando  : {len(missing)}")

    if missing:
        print("\n  ⚠ Períodos faltando:")
        MESES_NOME = ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"]
        for y, m in missing:
            print(f"    → {MESES_NOME[m-1]}/{y}  ({y}-{m:02d})")
    else:
        print("  ✓ Nenhum mês faltando!")

    # 4. Consolidar série diária SP
    print("\n[4/5] Consolidando série diária para SP...")

    if not all_daily:
        print("  [!] Nenhum dado válido para consolidar.")
        return

    df_all_daily = pd.concat(all_daily, ignore_index=True)
    print(f"  Registros diários brutos (todas as estações): {len(df_all_daily):,}")

    # Deduplicar caso haja sobreposição entre arquivos
    group_cols = ["cod_estacao", "date"] if "cod_estacao" in df_all_daily.columns else ["date"]
    df_all_daily = (df_all_daily
                    .sort_values(group_cols)
                    .drop_duplicates(subset=group_cols, keep="last"))
    df_all_daily, excluded = exclude_station_days(df_all_daily, exclusions)
    excluded.to_csv(Path(OUTPUT_REPORT).with_name("cemaden_estacoes_dias_excluidos.csv"), index=False)
    print(f"  Após deduplicação                           : {len(df_all_daily):,}")

    # Salvar metadados das estações
    if "cod_estacao" in df_all_daily.columns:
        stations = (df_all_daily
                    .groupby("cod_estacao")[["nome_estacao","lat","lon"]]
                    .first()
                    .reset_index()
                    .dropna(subset=["lat","lon"]))
        stations["dist_sp_km"] = stations.apply(
            lambda r: haversine(r["lat"], r["lon"], SP_LAT, SP_LON), axis=1
        )
        stations.to_csv(OUTPUT_STATIONS, index=False)
        print(f"  Estações únicas: {len(stations)}  → '{OUTPUT_STATIONS}'")

    # Série SP
    df_sp = build_sp_series(df_all_daily)

    # Preencher range completo
    all_days = pd.DataFrame({"date": pd.date_range(
        f"{START_PERIOD[0]}-{START_PERIOD[1]:02d}-01",
        pd.Timestamp(END_PERIOD[0], END_PERIOD[1], 1) + pd.offsets.MonthEnd(0)
    )})
    df_sp = pd.merge(all_days, df_sp, on="date", how="left")

    # Lags e acumulados (consistente com build_dataset.py e open_meteo)
    df_sp = df_sp.sort_values("date").reset_index(drop=True)
    for lag in [1, 2, 3]:
        df_sp[f"cemaden_precip_lag{lag}d"] = df_sp["cemaden_precip_idw_mm"].shift(lag) \
            if "cemaden_precip_idw_mm" in df_sp.columns \
            else df_sp["cemaden_precip_mean_mm"].shift(lag)
    for w in [3, 7, 14]:
        base = "cemaden_precip_idw_mm" if "cemaden_precip_idw_mm" in df_sp.columns \
               else "cemaden_precip_mean_mm"
        df_sp[f"cemaden_precip_acc{w}d"] = df_sp[base].rolling(w).sum()

    df_sp.to_csv(OUTPUT_DAILY, index=False)

    n_ok  = df_sp[df_sp.columns[1]].notna().sum()
    n_gap = df_sp[df_sp.columns[1]].isna().sum()
    print(f"\n  ✓ Série SP salva em '{OUTPUT_DAILY}'")
    print(f"    Shape       : {df_sp.shape}")
    print(f"    Dias com dados : {n_ok}")
    print(f"    Dias sem dados : {n_gap}")

    # 5. Relatório de gaps
    print("\n[5/5] Gerando relatório de qualidade...")
    _write_report(cat_df, missing, df_all_daily, df_sp, errors)
    print(f"  Relatório → '{OUTPUT_REPORT}'")
    print("\n" + "=" * 65)
    print("✓ Consolidação concluída!")
    print("=" * 65)


def _write_report(catalog: pd.DataFrame, missing: list,
                  df_daily: pd.DataFrame, df_sp: pd.DataFrame,
                  errors: list):
    MESES_NOME = ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"]
    lines = []

    lines += [
        "=" * 65,
        "RELATÓRIO DE QUALIDADE — CEMADEN SP",
        f"Gerado em: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}",
        "=" * 65,
        "",
        "── RESUMO ────────────────────────────────────────────────────",
        f"Arquivos processados   : {len(catalog)}",
        f"Leituras brutas válidas: {int(catalog['linhas'].sum()):,}",
        f"Registros estação-dia após exclusões: {len(df_daily):,}",
        f"Estações únicas        : {df_daily['cod_estacao'].nunique() if 'cod_estacao' in df_daily.columns else '?'}",
        f"Meses esperados        : {len(list(all_months_in_range(START_PERIOD, END_PERIOD)))}",
        f"Meses faltando         : {len(missing)}",
        f"Arquivos com erro      : {len(errors)}",
        "Exclusões estação-dia: cemaden_estacoes_dias_excluidos.csv",
        "cemaden_intensity_max_hor: nome legado; máximo por leitura, não intensidade horária.",
        "Fuso: hipótese histórica de horário local mantida; origem dos timestamps requer validação.",
        "",
    ]

    if missing:
        lines += ["── PERÍODOS FALTANDO ─────────────────────────────────────────"]
        for y, m in missing:
            lines.append(f"  {MESES_NOME[m-1]}/{y}  ({y}-{m:02d})")
        lines += [
            "",
            "  Para baixar os meses faltando:",
            "  URL: http://www2.cemaden.gov.br/mapainterativo/download/downpluv.php",
            "  UF: SP | Município: São Paulo | Selecione o mês acima",
            "",
        ]
    else:
        lines += ["── COBERTURA TEMPORAL ────────────────────────────────────────",
                  "  ✓ Todos os meses do período estão presentes.", ""]

    lines += ["── CATÁLOGO DE ARQUIVOS ──────────────────────────────────────"]
    for _, row in catalog.sort_values(["ano","mes"]).iterrows():
        flag = "✓" if pd.notna(row["ano"]) else "?"
        lines.append(
            f"  {flag} {row['arquivo']:35s} | {row['ano']}-{int(row['mes']):02d} "
            f"| {row['linhas']:>8,} leituras | {row['n_estacoes']} estações"
            f" | fonte: {row['periodo_fonte']}"
        )

    if errors:
        lines += ["", "── ERROS ─────────────────────────────────────────────────────"]
        for fname, msg in errors:
            lines.append(f"  ✗ {fname}: {msg}")

    lines += [
        "",
        "── QUALIDADE DA SÉRIE DIÁRIA SP ──────────────────────────────",
    ]
    base_col = "cemaden_precip_idw_mm" if "cemaden_precip_idw_mm" in df_sp.columns \
               else "cemaden_precip_mean_mm"
    if base_col in df_sp.columns:
        lines += [
            f"  Dias com dados : {df_sp[base_col].notna().sum()}",
            f"  Dias sem dados : {df_sp[base_col].isna().sum()}",
            f"  Média diária   : {df_sp[base_col].mean():.2f} mm",
            f"  Máximo diário  : {df_sp[base_col].max():.1f} mm",
        ]
        # Meses com dados faltando
        df_sp2 = df_sp.copy()
        df_sp2["year_month"] = df_sp2["date"].dt.to_period("M")
        missing_months_data = (df_sp2.groupby("year_month")[base_col]
                               .apply(lambda x: x.isna().mean()))
        parcial = missing_months_data[missing_months_data > 0.5]
        if not parcial.empty:
            lines += ["", "  Meses com >50% dados faltando:"]
            for ym, pct in parcial.items():
                lines.append(f"    {ym}: {pct*100:.0f}% faltando")

    lines.append("")
    lines.append("=" * 65)

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    try:
        from dateutil.relativedelta import relativedelta
    except ImportError:
        os.system("pip install python-dateutil --break-system-packages -q")
        from dateutil.relativedelta import relativedelta

    main()
