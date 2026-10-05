"""Tratamento reproduzível de GeoSampa + associação dos registros CGE.

Executar da raiz: python scripts/etl/transform_geosampa.py
Lê os ZIPs sem extrair nem alterar os originais. As medidas usam EPSG:31983.
Saídas são descritivas: snapshots urbanos não são automaticamente históricos.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import zipfile
import unicodedata

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

ROOT = Path(__file__).resolve().parents[2]
CRS = 31983
ALIASES = {
    "ARICANDUVAVILAFORMOSA": "ARICANDUVAFORMOSACARRAO",
    "CASAVERDE": "CASAVERDELIMAOCACHOEIRINHA",
    "CASAVERDECACHOEIRINHA": "CASAVERDELIMAOCACHOEIRINHA",
    "FREGUESIADOO": "FREGUESIABRASILANDIA",
    "SANTANA": "SANTANATUCURUVI",
    "PERUS": "PERUSANHANGUERA",
    "SAOMIGUELPAULISTA": "SAOMIGUEL",
}


def normalize(value):
    text = unicodedata.normalize("NFD", str(value).upper())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"[^A-Z0-9]", "", text)


def projected(frame):
    if frame.crs is None:
        raise ValueError("Camada sem CRS: não é seguro presumir coordenadas")
    return frame.to_crs(CRS)


def polygons(geometry):
    """Extrai apenas componentes poligonais após make_valid."""
    if geometry.geom_type in ("Polygon", "MultiPolygon"):
        return geometry
    parts = [polygons(g) for g in getattr(geometry, "geoms", [])]
    return shapely.union_all([g for g in parts if not g.is_empty])


def clean_geometry(frame, audit, label, polygon=False):
    frame = projected(frame).copy()
    if frame.geometry.isna().any() or frame.geometry.is_empty.any():
        raise ValueError(f"{label}: geometria ausente/vazia; revisar antes de agregar")
    invalid = ~frame.geometry.is_valid
    audit[label] = {"registros": len(frame), "geometrias_reparadas": int(invalid.sum())}
    if invalid.any():
        frame.loc[invalid, frame.geometry.name] = frame.loc[invalid].geometry.make_valid()
    if polygon:
        other = ~frame.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
        frame.loc[other, frame.geometry.name] = frame.loc[other].geometry.map(polygons)
    if frame.geometry.is_empty.any() or not frame.geometry.is_valid.all():
        raise ValueError(f"{label}: reparo não gerou geometrias válidas")
    return frame


def read_boundaries(path, audit):
    frame = gpd.read_file(path, layer="subprefeitura_v2")
    frame = clean_geometry(frame, audit, "subprefeituras", polygon=True)
    frame = frame.rename(columns={"cd_subprefeitura": "codigo_subprefeitura",
                                  "nm_subprefeitura": "nome_subprefeitura"})
    frame["codigo_subprefeitura"] = frame.codigo_subprefeitura.astype(str).str.zfill(2)
    if len(frame) != 32 or not frame.codigo_subprefeitura.is_unique:
        raise ValueError("Esperadas 32 subprefeituras com códigos únicos")
    frame["area_km2"] = frame.geometry.area / 1e6
    union = shapely.union_all(frame.geometry)
    overlap = float(frame.geometry.area.sum() - union.area)
    audit["subprefeituras"]["sobreposicao_m2"] = overlap
    if overlap > 100:  # Tolerância explícita para resíduos de fronteira no município inteiro.
        raise ValueError(f"Limites se sobrepõem em {overlap:.1f} m²; revisar fronteiras")
    return frame.sort_values("codigo_subprefeitura").reset_index(drop=True)


def map_cge(raw, boundaries):
    official = {normalize(row.nome_subprefeitura): (row.codigo_subprefeitura, row.nome_subprefeitura)
                for row in boundaries.itertuples()}
    rows = []
    for name in sorted(raw.neighbor.dropna().unique()):
        key = normalize(name)
        mapped = ALIASES.get(key, key)
        if mapped not in official:
            raise ValueError(f"Nome CGE sem correspondência explícita: {name}")
        code, official_name = official[mapped]
        rows.append({"neighbor": name, "codigo_subprefeitura": code,
                     "nome_subprefeitura": official_name,
                     "regra": "alias_explicito" if key != mapped else "normalizacao_grafica"})
    mapping = pd.DataFrame(rows)
    if not mapping.codigo_subprefeitura.is_unique:
        raise ValueError("Mais de um rótulo CGE para a mesma subprefeitura; revisar")
    return mapping


def daily_cge(raw, mapping, codes):
    """0 significa sem registro, nunca ausência física confirmada de alagamento."""
    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw.date, format="%d-%m-%Y", errors="raise")
    if (raw.neighbor.isna() & raw[["current_zone", "status", "way"]].notna().any(axis=1)).any():
        raise ValueError("Há ocorrência CGE sem subprefeitura")
    events = raw[raw.neighbor.notna()].merge(mapping, on="neighbor", how="left", validate="many_to_one")
    if events.codigo_subprefeitura.isna().any():
        raise ValueError("Ocorrência sem código geográfico")
    counts = events.groupby(["date", "codigo_subprefeitura"]).size().rename("n_registros_cge")
    index = pd.MultiIndex.from_product([pd.date_range(raw.date.min(), raw.date.max()), codes],
                                       names=["date", "codigo_subprefeitura"])
    panel = counts.reindex(index).reset_index()
    panel["dia_presente_base_cge"] = panel.date.isin(raw.date)
    observed = panel.dia_presente_base_cge
    panel.loc[observed, "n_registros_cge"] = panel.loc[observed, "n_registros_cge"].fillna(0)
    panel["n_registros_cge"] = panel.n_registros_cge.astype("Int64")
    panel["tem_registro_cge"] = panel.n_registros_cge.gt(0).astype("Int64")
    panel["coleta_confirmada"] = pd.Series(pd.NA, index=panel.index, dtype="boolean")
    return panel


def clip_to_regions(frame, boundaries):
    """Recorta cada feição nas fronteiras reais, inclusive quando cruza regiões."""
    for row in boundaries.itertuples():
        boundary = getattr(row, boundaries.geometry.name)
        hits = frame.sindex.query(boundary, predicate="intersects")
        if not len(hits):
            continue
        part = frame.iloc[hits].copy()
        part.geometry = shapely.intersection(part.geometry.array, boundary)
        part = part.loc[~part.geometry.is_empty].copy()
        if len(part):
            yield row.codigo_subprefeitura, part


def vegetation(paths, boundaries, audit):
    regional = defaultdict(list)
    categories = defaultdict(list)
    records = []
    attribute_counts = []
    found_codes = set()
    flight_years = set()
    for i, path in enumerate(paths, 1):
        print(f"Vegetação {i}/{len(paths)}: {path.name}", flush=True)
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ValueError(f"ZIP corrompido: {path}")
            names = archive.namelist()
            for ext in (".shp", ".shx", ".dbf", ".prj"):
                if sum(n.lower().endswith(ext) for n in names) != 1:
                    raise ValueError(f"Componente {ext} ausente/ambíguo: {path}")
        shp = next(n for n in names if n.lower().endswith(".shp"))
        frame = gpd.read_file(f"/vsizip/{path.resolve().as_posix()}/{shp}")
        frame = clean_geometry(frame, audit, path.name, polygon=True)
        frame["categoria"] = pd.to_numeric(frame.cd_categ, errors="raise").astype(int)
        if not frame.categoria.between(1, 15).all():
            raise ValueError("Categoria de vegetação fora do catálogo 1–15")
        codes = set(frame.cd_subpref.astype(str).str.zfill(2))
        if not codes <= set(boundaries.codigo_subprefeitura):
            raise ValueError("Código de vegetação inexistente nos limites")
        found_codes |= codes
        flight_years.update(frame.tx_dt_voo.dropna().astype(str))
        for (category, description), n in frame.groupby(["categoria", "tx_ct_sbct"], dropna=False).size().items():
            attribute_counts.append({"arquivo": path.name, "categoria": int(category),
                                     "descricao_original": str(description), "registros": int(n)})
        area_before = float(frame.geometry.area.sum())
        clipped_area = 0.0
        for code, part in clip_to_regions(frame, boundaries):
            clipped_area += float(part.geometry.area.sum())
            union = shapely.union_all(part.geometry)
            regional[code].append(union)
            for category, group in part.groupby("categoria"):
                categories[(code, int(category))].append(shapely.union_all(group.geometry))
        records.append({"arquivo": path.name, "registros": len(frame),
                        "area_somada_original_m2": area_before,
                        "area_somada_recortada_m2": clipped_area})
    if found_codes != set(boundaries.codigo_subprefeitura):
        raise ValueError("Arquivos de vegetação não cobrem todos os códigos de subprefeitura")
    if flight_years != {"2017"}:
        raise ValueError(f"Anos de voo diferentes dos esperados: {flight_years}")
    areas = dict(zip(boundaries.codigo_subprefeitura, boundaries.geometry.area))
    features, by_category = [], []
    for code, area in areas.items():
        union_area = float(shapely.union_all(regional[code]).area)
        class_sum = 0.0
        for category in range(1, 16):
            class_area = float(shapely.union_all(categories[(code, category)]).area)
            class_sum += class_area
            by_category.append({"codigo_subprefeitura": code, "categoria": category,
                                "area_uniao_m2": class_area, "percentual_area_subprefeitura": 100*class_area/area})
        features.append({"codigo_subprefeitura": code, "area_mapeada_vegetacao_m2": union_area,
                         "pct_area_mapeada_vegetacao": 100*union_area/area,
                         "sobreposicao_entre_categorias_m2": max(0.0, class_sum-union_area),
                         "ano_voo_vegetacao": 2017})
    return pd.DataFrame(features), pd.DataFrame(by_category), pd.DataFrame(records), pd.DataFrame(attribute_counts)


def drainage(path, boundaries, audit):
    frame = clean_geometry(gpd.read_file(path, layer="drenagem"), audit, "drenagem")
    if not frame.geometry.geom_type.isin(["LineString", "MultiLineString"]).all():
        raise ValueError("Drenagem contém geometrias que não são linhas")
    frame["tipo"] = frame.nm_tipo_curso_hidrografia.fillna("SEM_CLASSIFICACAO")
    dates = sorted(frame.dt_atualizacao.dropna().astype(str).unique())
    audit["drenagem"]["datas_atualizacao"] = dates
    rows, details = [], []
    types = sorted(frame.tipo.unique())
    for code, part in clip_to_regions(frame, boundaries):
        # União remove segmentos coincidentes, inclusive duplicatas exatas.
        km = float(shapely.union_all(part.geometry).length / 1000)
        area = float(boundaries.loc[boundaries.codigo_subprefeitura.eq(code), "area_km2"].iloc[0])
        rows.append({"codigo_subprefeitura": code, "rede_mapeada_km": km,
                     "densidade_rede_km_por_km2": km/area,
                     "trechos_intersectando": len(part)})
        for kind in types:
            group = part.loc[part.tipo.eq(kind)]
            length = float(shapely.union_all(group.geometry).length/1000)
            details.append({"codigo_subprefeitura": code, "tipo": kind,
                            "comprimento_uniao_km": length, "densidade_km_por_km2": length/area})
    result = boundaries[["codigo_subprefeitura"]].merge(pd.DataFrame(rows), how="left", validate="one_to_one")
    result[["rede_mapeada_km", "densidade_rede_km_por_km2", "trechos_intersectando"]] = result[
        ["rede_mapeada_km", "densidade_rede_km_por_km2", "trechos_intersectando"]].fillna(0)
    result["atualizacao_cadastro_drenagem"] = " | ".join(dates)
    audit["drenagem"]["soma_comprimento_regioes_km"] = float(result.rede_mapeada_km.sum())
    city = shapely.union_all(boundaries.geometry)
    clipped = shapely.intersection(frame.geometry.array, city)
    audit["drenagem"]["comprimento_uniao_municipal_km"] = float(shapely.union_all(clipped).length/1000)
    audit["drenagem"]["nota_fronteiras"] = "Trecho coincidente com fronteira pode participar das duas regiões; não somar para obter total municipal."
    return result, pd.DataFrame(details)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=ROOT / "data/input/geosampa")
    parser.add_argument("--cge", type=Path, default=ROOT / "data/outputs/datasets/final_data.csv")
    parser.add_argument("--saida", type=Path, default=ROOT / "data/outputs/geosampa")
    args = parser.parse_args()
    audit = {}
    boundary_path = args.entrada / "geoportal_subprefeitura_v2.gpkg"
    drainage_path = args.entrada / "geoportal_drenagem.gpkg"
    zips = sorted(args.entrada.glob("SIRGAS_SHP_vegetacao_pmd_*.zip"))
    if len(zips) != 32:
        raise ValueError(f"Esperados 32 ZIPs de vegetação; encontrados {len(zips)}")
    paths = [boundary_path, drainage_path, args.cge, *zips]
    source_hashes = {str(p.resolve()): sha256(p) for p in paths}
    boundaries = read_boundaries(boundary_path, audit)
    raw = pd.read_csv(args.cge, dtype=str)
    mapping = map_cge(raw, boundaries)
    panel = daily_cge(raw, mapping, boundaries.codigo_subprefeitura)
    veg, categories, files, descriptions = vegetation(zips, boundaries, audit)
    print("Recortando rede de drenagem...", flush=True)
    drain, drain_types = drainage(drainage_path, boundaries, audit)
    features = boundaries[["codigo_subprefeitura", "nome_subprefeitura", "area_km2"]].merge(
        veg, validate="one_to_one").merge(drain, validate="one_to_one")
    wide = categories.pivot(index="codigo_subprefeitura", columns="categoria", values="percentual_area_subprefeitura")
    wide.columns = [f"pct_area_categoria_vegetacao_{int(c):02d}" for c in wide.columns]
    features = features.merge(wide, on="codigo_subprefeitura", validate="one_to_one")
    features["referencia_limites"] = "cadastro GeoSampa subprefeitura_v2 baixado em 2026"
    features["uso_temporal"] = "snapshot_descritivo_nao_validado_como_historico"
    if not features.pct_area_mapeada_vegetacao.between(0, 100.000001).all():
        raise ValueError("Área mapeada excede a área regional")
    if features.select_dtypes(include="number").isna().any().any():
        raise ValueError("Indicadores numéricos incompletos")
    audit["cge"] = {"linhas": len(raw), "duplicatas_excedentes_preservadas": int(raw.duplicated().sum()),
                    "rotulos_mapeados": len(mapping), "linhas_painel": len(panel),
                    "registros_antes": int(raw.neighbor.notna().sum()),
                    "registros_depois": int(panel.n_registros_cge.sum())}
    if audit["cge"]["registros_antes"] != audit["cge"]["registros_depois"]:
        raise ValueError("Integração perdeu registros CGE")
    # Não publicar saídas parciais de etapas que ainda não foram validadas.
    args.saida.mkdir(parents=True, exist_ok=True)
    outputs = {"correspondencia_cge_subprefeituras.csv": mapping,
               "indicadores_subprefeituras.csv": features,
               "vegetacao_por_categoria.csv": categories,
               "auditoria_arquivos_vegetacao.csv": files,
               "auditoria_categorias_vegetacao.csv": descriptions,
               "drenagem_por_tipo.csv": drain_types,
               "cge_subprefeitura_dia.csv": panel}
    for name, frame in outputs.items():
        frame.to_csv(args.saida / name, index=False)
    boundaries.to_file(args.saida / "subprefeituras_tratadas.gpkg", layer="subprefeituras", driver="GPKG")
    audit["manifesto"] = {"execucao_utc": datetime.now(timezone.utc).isoformat(), "epsg": CRS,
        "hashes_entrada_sha256": source_hashes,
        "versoes": {p: importlib.metadata.version(p) for p in ("pandas", "geopandas", "shapely", "pyogrio", "pyproj")},
        "restricao": "Não houve join dos snapshots urbanos ao histórico diário: disponibilidade histórica não comprovada.",
        "hash_script": sha256(Path(__file__))}
    (args.saida / "auditoria.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Concluído: {len(features)} regiões; {len(panel)} linhas CGE região-dia. Saída: {args.saida}")


if __name__ == "__main__":
    main()
