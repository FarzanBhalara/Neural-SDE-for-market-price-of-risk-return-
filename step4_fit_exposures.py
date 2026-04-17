import os

import numpy as np

from data import load_panel_artifact
from models.exposures import build_beta_mask, clip_beta, compute_rolling_market_beta, smooth_market_beta


PANEL_FILE = "outputs/panel_step1.npz"
EXPOSURES_FILE = "outputs/step4_exposures.npz"
STEP2_BETA_FILE = "outputs/step2_beta_market_tmp.npz"

BETA_WINDOW = 60
BETA_HALFLIFE = 25
BETA_SHRINK_TARGET = 1.0
BETA_SHRINK_WEIGHT = 0.15
BETA_CLIP = 5.0


def main():
    os.makedirs("outputs", exist_ok=True)
    panel = load_panel_artifact(PANEL_FILE)
    if os.path.exists(STEP2_BETA_FILE):
        beta_artifact = np.load(STEP2_BETA_FILE, allow_pickle=True)
        beta_market = beta_artifact["beta_market"].astype(np.float32)
        beta_valid_mask = beta_artifact["beta_valid_mask"].astype(bool)
        beta_source = STEP2_BETA_FILE
    else:
        beta_raw, beta_valid = compute_rolling_market_beta(
            excess_return=panel["excess_return"],
            market_return=panel["market_excess_return"],
            valid_mask=panel["lookback_60_mask"].astype(bool),
            window=BETA_WINDOW,
        )
        beta_smooth = smooth_market_beta(
            beta=beta_raw,
            valid_mask=beta_valid,
            halflife=BETA_HALFLIFE,
            shrink_target=BETA_SHRINK_TARGET,
            shrink_weight=BETA_SHRINK_WEIGHT,
        )
        beta_market = clip_beta(beta_smooth, lower=-BETA_CLIP, upper=BETA_CLIP)
        beta_valid_mask = build_beta_mask(beta_market, beta_valid)
        beta_source = "recomputed"

    np.savez(
        EXPOSURES_FILE,
        dates=panel["dates"],
        asset_ids=panel["asset_ids"],
        beta_market=beta_market.astype(np.float32),
        beta_valid_mask=beta_valid_mask.astype(bool),
        beta_window=np.asarray([BETA_WINDOW], dtype=np.int32),
        beta_halflife=np.asarray([BETA_HALFLIFE], dtype=np.int32),
        beta_shrink_target=np.asarray([BETA_SHRINK_TARGET], dtype=np.float32),
        beta_shrink_weight=np.asarray([BETA_SHRINK_WEIGHT], dtype=np.float32),
        beta_clip=np.asarray([BETA_CLIP], dtype=np.float32),
    )

    valid_beta = beta_market[beta_valid_mask]
    print("Step 4 complete: rolling market betas fit.")
    print(f"Beta source: {beta_source}")
    print(f"Saved exposures to {EXPOSURES_FILE}")
    print(f"Valid beta rows: {int(beta_valid_mask.sum())}")
    print(f"Beta mean: {float(np.mean(valid_beta)):.4f}")
    print(f"Beta std: {float(np.std(valid_beta)):.4f}")


if __name__ == "__main__":
    main()
