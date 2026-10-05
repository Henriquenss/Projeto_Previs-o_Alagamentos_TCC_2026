"""Exporta casos para revisão; não elimina eventos nem modifica os datasets."""
import pandas as pd
from build_integrated_dataset import BASE


def main():
    report = BASE / "relatórios"
    report.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(BASE / "final_data.csv", dtype=str)
    groups = raw.groupby(list(raw.columns), dropna=False).size().reset_index(name="repeticoes")
    groups = groups[groups.repeticoes > 1]
    groups.to_csv(report / "cge_repeticoes_para_revisao.csv", index=False)
    daily = pd.read_csv(BASE / "cemaden_sp_diario.csv")
    # Limiar apenas de triagem, não critério automático de exclusão.
    daily.loc[daily.cemaden_precip_max_mm > 500,
              ["date", "cemaden_precip_idw_mm", "cemaden_precip_max_mm"]].to_csv(
                  report / "cemaden_extremos_para_revisao.csv", index=False)
    print(f"CGE: {len(groups)} grupos, {int((groups.repeticoes - 1).sum())} repetições excedentes; mantidas")


if __name__ == "__main__":
    main()
