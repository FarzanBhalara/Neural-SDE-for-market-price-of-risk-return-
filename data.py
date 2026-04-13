from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_DIR = Path("outputs")
PANEL_FILE = OUTPUT_DIR / "panel_step1.npz"
PANEL_SUMMARY_CSV = OUTPUT_DIR / "panel_step1_summary.csv"
COVARIANCE_FILE = OUTPUT_DIR / "step2_covariance_predictions.npz"
EXPOSURES_FILE = OUTPUT_DIR / "step4_exposures.npz"
MARKET_PARAMS_FILE = OUTPUT_DIR / "step6_market_params_panel.npz"


def load_panel_artifact(path=PANEL_FILE):
    npz = np.load(path, allow_pickle=True)
    return {key: npz[key] for key in npz.files}


def load_panel_summary(path=PANEL_SUMMARY_CSV):
    return pd.read_csv(path, parse_dates=["date"])


def load_covariance_predictions(path=COVARIANCE_FILE):
    npz = np.load(path, allow_pickle=True)
    return {key: npz[key] for key in npz.files}


def load_exposures(path=EXPOSURES_FILE):
    npz = np.load(path, allow_pickle=True)
    return {key: npz[key] for key in npz.files}


def load_market_params(path=MARKET_PARAMS_FILE):
    npz = np.load(path, allow_pickle=True)
    return {key: npz[key] for key in npz.files}
