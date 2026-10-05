"""Testes locais: python -m unittest discover -s scripts/etl -p test_*.py"""
import unittest
import json
from pathlib import Path
import pandas as pd
import numpy as np
from transforming_cemaden_data import exclude_station_days, build_sp_series
from extract_open_meteo_forecast_data import aggregate_to_daily
from build_integrated_dataset import build, BASE, CEMADEN_COLS


class CorrectionTests(unittest.TestCase):
    def test_missing_and_partial_precipitation_are_not_zero(self):
        hourly = pd.DataFrame({"datetime": pd.date_range("2024-01-01", periods=72, freq="h"),
                               "precipitation": [np.nan]*24 + [0.0]*24 + [1.0]*23 + [np.nan]})
        result = aggregate_to_daily(hourly, {"precipitation": "sum"}, [])
        self.assertTrue(pd.isna(result.precipitation.iloc[0]))
        self.assertEqual(result.precipitation.iloc[1], 0.0)
        self.assertTrue(pd.isna(result.precipitation.iloc[2]))

    def test_duplicate_hours_rejected(self):
        hourly = pd.DataFrame({"datetime": pd.to_datetime(["2024-01-01"]*24),
                               "precipitation": [1.0]*24})
        with self.assertRaises(ValueError):
            aggregate_to_daily(hourly, {"precipitation": "sum"}, [])

    def test_exclusion_keeps_other_station_and_renormalizes(self):
        daily = pd.DataFrame({"cod_estacao": ["bad", "good", "bad"],
                              "date": pd.to_datetime(["2021-03-04", "2021-03-04", "2021-03-05"]),
                              "precip_total_mm": [111172.18, 10.0, 1.0],
                              "precip_max_hor_mm": [111172.18, 5.0, 1.0],
                              "lat": [-23.603, -23.407, -23.603],
                              "lon": [-46.449, -46.753, -46.449]})
        exclusions = pd.DataFrame({"cod_estacao": ["bad"],
                                   "date": pd.to_datetime(["2021-03-04"]), "motivo": ["erro"]})
        kept, excluded = exclude_station_days(daily, exclusions)
        self.assertEqual(len(excluded), 1)
        self.assertEqual(len(kept), 2)
        result = build_sp_series(kept)
        self.assertAlmostEqual(result.cemaden_precip_idw_mm.iloc[0], 10.0)
        self.assertAlmostEqual(result.cemaden_precip_idw_mm.iloc[1], 1.0)

    def test_calendar_and_integrated_sources(self):
        integrated = pd.read_csv(BASE / "dataset_final_integrado.csv", parse_dates=["date"])
        pd.testing.assert_frame_equal(integrated, build(), check_dtype=False, atol=1e-9)
        expected = pd.date_range("2015-01-01", "2026-05-31")
        self.assertEqual(list(integrated.date), list(expected))
        self.assertFalse(integrated[CEMADEN_COLS[:5]].isna().any().any())
        self.assertEqual(integrated.has_flooding.sum(), 1031)

    def test_lags_and_rolling_windows(self):
        df = pd.read_csv(BASE / "cemaden_sp_diario.csv")
        values = df.cemaden_precip_idw_mm.to_numpy()
        for day in ("2021-01-20", "2021-03-04", "2026-05-31"):
            i = df.index[df.date.eq(day)][0]
            for n in (1, 2, 3):
                self.assertAlmostEqual(df.at[i, f"cemaden_precip_lag{n}d"], values[i-n])
            for n in (3, 7, 14):
                self.assertAlmostEqual(df.at[i, f"cemaden_precip_acc{n}d"], sum(values[i-n+1:i+1]))

    def test_saved_forecasts_match_hourly_evidence(self):
        from repair_previous_runs import CACHE
        payload = json.loads(CACHE.read_text(encoding="utf-8"))
        hourly = pd.DataFrame(payload["hourly"]).rename(columns={"time": "datetime"})
        hourly["datetime"] = pd.to_datetime(hourly["datetime"])
        saved = pd.read_csv(BASE / "open_meteo_previous_runs_sp.csv").set_index("date")
        for lead in (1, 2, 3):
            col = f"precipitation_previous_day{lead}"
            for day, group in hourly.groupby(hourly.datetime.dt.strftime("%Y-%m-%d")):
                valid = group[col].dropna()
                actual = saved.at[day, col]
                if len(valid) < 24:
                    self.assertTrue(pd.isna(actual))
                else:
                    self.assertAlmostEqual(actual, sum(valid))
            expected_error = saved[col] - saved.precipitation
            np.testing.assert_allclose(saved[f"precipitation_error_day{lead}"], expected_error,
                                       rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
