"""Confere as saídas reais: python scripts/etl/validate_geosampa.py.

Não altera arquivos. Verifica chaves, reconciliação do CGE, medidas e hashes
das entradas registradas. Execute após transform_geosampa.py.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from transform_geosampa import ROOT, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saida", type=Path, default=ROOT / "data/outputs/geosampa")
    args = parser.parse_args()
    def read(name):
        return pd.read_csv(args.saida / name, dtype={"codigo_subprefeitura": str})
    audit = json.loads((args.saida / "auditoria.json").read_text(encoding="utf-8"))
    features = read("indicadores_subprefeituras.csv")
    mapping = read("correspondencia_cge_subprefeituras.csv")
    panel = read("cge_subprefeitura_dia.csv")
    categories = read("vegetacao_por_categoria.csv")
    codes = set(features.codigo_subprefeitura)
    assert len(features) == len(codes) == 32
    assert set(mapping.codigo_subprefeitura) == codes
    assert not panel.duplicated(["date", "codigo_subprefeitura"]).any()
    assert panel.groupby("date").codigo_subprefeitura.nunique().eq(32).all()
    assert panel.coleta_confirmada.isna().all()
    present = panel.dia_presente_base_cge.eq(True)
    assert panel.loc[~present, "tem_registro_cge"].isna().all()
    assert panel.loc[present, "tem_registro_cge"].eq(panel.loc[present, "n_registros_cge"].gt(0)).all()
    assert int(panel.n_registros_cge.sum()) == audit["cge"]["registros_antes"]
    assert features.area_km2.gt(0).all()
    assert features.pct_area_mapeada_vegetacao.between(0, 100.000001).all()
    assert features.rede_mapeada_km.ge(0).all()
    np.testing.assert_allclose(features.densidade_rede_km_por_km2,
                               features.rede_mapeada_km / features.area_km2)
    assert categories.groupby("codigo_subprefeitura").categoria.nunique().eq(15).all()
    assert categories.percentual_area_subprefeitura.between(0, 100.000001).all()
    for name, expected in audit["manifesto"]["hashes_entrada_sha256"].items():
        assert sha256(Path(name)) == expected, f"Entrada mudou desde o processamento: {name}"
    print(f"Validação OK: {len(features)} regiões, {len(panel)} linhas região-dia, "
          f"{int(panel.n_registros_cge.sum())} registros CGE conservados.")
    print("Todas as entradas mantêm o hash registrado; nenhuma foi modificada durante o tratamento.")
    print(features[["pct_area_mapeada_vegetacao", "densidade_rede_km_por_km2"]].agg(["min", "max"]).to_string())


if __name__ == "__main__":
    main()
