"""Testes: python -m unittest discover -s scripts/etl -p test_geosampa.py"""
import unittest
import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry import box, LineString, Polygon
from transform_geosampa import (CRS, clean_geometry, clip_to_regions,
                                map_cge, daily_cge, projected)


class GeoSampaTests(unittest.TestCase):
    def regions(self):
        return gpd.GeoDataFrame({"codigo_subprefeitura": ["01", "02"],
                                 "nome_subprefeitura": ["PERUS-ANHANGUERA", "SANTANA-TUCURUVI"]},
                                geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10)], crs=CRS)

    def test_cross_border_polygon_and_overlap(self):
        # Um polígono com 100 m² é dividido em 50 m² de cada lado.
        # A duplicata não pode dobrar a área do indicador.
        data = gpd.GeoDataFrame(geometry=[box(5, 0, 15, 10)]*2, crs=CRS)
        pieces = dict(clip_to_regions(data, self.regions()))
        self.assertEqual(set(pieces), {"01", "02"})
        for piece in pieces.values():
            self.assertAlmostEqual(shapely.union_all(piece.geometry).area, 50)

    def test_line_crosses_border(self):
        data = gpd.GeoDataFrame(geometry=[LineString([(0, 5), (20, 5)])], crs=CRS)
        pieces = dict(clip_to_regions(data, self.regions()))
        self.assertAlmostEqual(pieces["01"].geometry.length.sum(), 10)
        self.assertAlmostEqual(pieces["02"].geometry.length.sum(), 10)

    def test_explicit_name_aliases(self):
        mapping = map_cge(pd.DataFrame({"neighbor": ["Perus", "Santana"]}), self.regions())
        self.assertEqual(mapping.codigo_subprefeitura.tolist(), ["01", "02"])
        with self.assertRaises(ValueError):
            map_cge(pd.DataFrame({"neighbor": ["Região desconhecida"]}), self.regions())

    def test_unobserved_day_is_not_negative(self):
        raw = pd.DataFrame({"date": ["01-01-2020", "01-01-2020", "03-01-2020"],
                            "neighbor": ["Perus", "Perus", None],
                            "current_zone": ["Norte", "Norte", None],
                            "status": ["ativo", "ativo", None], "way": ["Rua A", "Rua A", None]})
        mapping = map_cge(raw, self.regions())
        panel = daily_cge(raw, mapping, ["01", "02"])
        self.assertEqual(len(panel), 6)
        self.assertEqual(panel.n_registros_cge.sum(), 2)  # Repetições preservadas.
        self.assertTrue(panel.loc[panel.date.eq("2020-01-02"), "tem_registro_cge"].isna().all())
        self.assertTrue(panel.loc[panel.date.eq("2020-01-03"), "tem_registro_cge"].eq(0).all())
        self.assertTrue(panel.coleta_confirmada.isna().all())

    def test_crs_required(self):
        with self.assertRaises(ValueError):
            projected(gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)]))

    def test_invalid_polygon_repaired_and_logged(self):
        frame = gpd.GeoDataFrame(geometry=[Polygon([(0,0), (2,2), (0,2), (2,0), (0,0)])], crs=CRS)
        audit = {}
        result = clean_geometry(frame, audit, "teste", polygon=True)
        self.assertTrue(result.geometry.is_valid.all())
        self.assertEqual(audit["teste"]["geometrias_reparadas"], 1)
        self.assertAlmostEqual(result.geometry.area.sum(), 2)


if __name__ == "__main__":
    unittest.main()
