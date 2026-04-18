# Discretized Neural SDE for Market Price of Risk and Return in a Multi-Asset Setting

This project presents a **discretized Neural SDE–inspired multi-asset pipeline** for modeling market volatility, a common time-varying risk-premium signal, and the resulting excess drift in a panel of NIFTY 50 assets.

The main idea of the project is:

- first build a clean **multi-asset panel** of prices and returns
- estimate the **volatility / covariance structure** of the market
- learn a **common scalar lambda signal** representing time-varying market-wide risk-premium conditions
- construct asset-level excess drift using the relation

  `mu_excess(i,t) ≈ sigma(i,t) × lambda(t)`

- evaluate whether the learned system captures volatility structure and market-premium regimes better than simple baselines

This project is strongest on the **sigma / covariance modeling stage**.  
The lambda stage is a **common market risk-premium proxy**, and the mu stage is a downstream constructed quantity that inherits the strengths of sigma and the limitations of lambda.

---

## Project Philosophy

The pipeline is built around a simple philosophy:

1. **Risk modeling comes first.**  
   Before estimating expected return or market price of risk, we first try to learn the volatility structure of the market in a multi-asset setting.

2. **Volatility should be modeled structurally.**  
   The project decomposes volatility into:
   - a **common market factor component**
   - an **idiosyncratic component**

3. **Risk premium is treated as a common market signal.**  
   Instead of learning a separate premium for every asset, the project uses a **shared scalar lambda(t)** that varies through time and reflects common market-wide premium conditions.

4. **Expected excess return is constructed from volatility and lambda.**  
   Once sigma and lambda are available, the project forms

   `mu_excess(i,t) ≈ sigma(i,t) × lambda(t)`

   as a simplified excess-drift estimate.

5. **Evaluation must be out of sample.**  
   The project uses a **chronological train/validation/test split** to reduce leakage and evaluate whether the model generalizes beyond training data.

---

## Important Note on Interpretation

This is **not** a full exact continuous-time Neural SDE implementation.

It is better described as a:

- **discretized Neural SDE–inspired framework**
- **multi-asset volatility and risk-premium modeling pipeline**
- **simplified one-factor / common-lambda formulation**

So the learned lambda should be interpreted as a:

- **common time-varying market risk-premium signal**
- or **common lambda proxy**

rather than a perfectly exact structural continuous-time market price of risk estimator.

---

## Repository Structure

### Main pipeline scripts

- `step0_fetch_nifty50_panel.py`  
  Fetches historical multi-asset price data.

- `step0_fetch_risk_free.py`  
  Fetches or constructs the daily risk-free rate series.

- `step0_resolve_nifty50_panel.py`  
  Resolves and prepares the NIFTY 50 panel data.

- `step1_preprocess_panel.py`  
  Builds the final multi-asset panel artifact, computes returns and excess returns, creates forward targets, and generates chronological train/validation/test splits.

- `step2_train_covariance.py`  
  Trains the volatility / covariance stage using factor and idiosyncratic components.

- `step3_evaluate_covariance.py`  
  Evaluates the learned covariance-driven sigma against baseline methods such as rolling volatility and EWMA.

- `step4_fit_exposures.py`  
  Fits exposures required for downstream drift construction.

- `step5_train_lambda.py`  
  Trains the common scalar lambda signal.

- `step6_export_market_params.py`  
  Exports final market parameters, including mu-related outputs.

### Supporting modules

- `data.py`  
  Utility functions for loading and saving panel artifacts and outputs.

- `models/`  
  Contains the modeling logic for:
  - panel processing
  - factor covariance
  - idiosyncratic volatility
  - volatility pipeline
  - lambda modeling
  - exposures
  - baselines
  - multivariate SDE-inspired components

### Data

The `data/` folder contains the cleaned input files needed by the pipeline, such as:
- multi-asset NIFTY 50 price panel
- index-level price series
- risk-free rate series

### Outputs

The `outputs/` folder contains the main final artifacts used for report interpretation, such as:
- covariance component plots
- lambda plots
- mu plots
- metrics CSVs

---

## Pipeline Flow

### Step 1 — Preprocessing and Panel Construction

This stage:
- loads the multi-asset NIFTY 50 panel
- aligns assets on a common date axis
- computes **log returns**
- computes **excess returns** by subtracting the daily risk-free rate
- builds forward targets such as:
  - future volatility targets
  - future excess mean return targets
- creates a **chronological train/validation/test split**

This is the foundation of the entire pipeline.

---

### Step 2 — Sigma / Covariance Modeling

This stage learns the volatility structure of the market.

The modeling idea is:

- estimate a **common market factor volatility**
- estimate **idiosyncratic volatility** for each asset
- combine them into a marginal sigma for each asset

Conceptually, volatility is decomposed into:
- **market-wide risk**
- **asset-specific risk**

This stage is the strongest part of the project.

---

### Step 3 — Covariance Evaluation

The learned sigma is evaluated against simpler baselines such as:

- rolling 20-day diagonal volatility
- EWMA diagonal volatility

Important evaluation metrics include:
- correlation with target sigma
- MAE
- RMSE
- QLIKE
- residual diagnostics

The project’s learned `covariance_sigma` outperforms the baseline volatility models on held-out test data, which suggests that sigma is learning meaningful out-of-sample structure rather than simply memorizing the training data.

---

### Step 4 — Exposure Fitting

This stage fits the exposures needed to connect the volatility structure to downstream quantities such as drift and risk premium.

---

### Step 5 — Lambda Modeling

This stage learns a **common scalar time-varying lambda(t)**.

Interpretation:
- positive lambda → more favorable market risk-premium regime
- low or negative lambda → weaker or adverse market-premium regime

The lambda stage is evaluated using:
- market-level predictive metrics
- train vs validation behavior
- regime-aware interpretation

The lambda signal is **meaningful but simplified**.  
It should be interpreted as a **common market risk-premium proxy**, not an exact structural estimator.

---

### Step 6 — Mu Construction

The final excess drift is constructed approximately as:

`mu_excess(i,t) ≈ sigma(i,t) × lambda(t)`

This means:
- sigma controls the asset-level scale
- lambda controls the common time-varying market premium regime

The resulting mu is useful, but it is still weaker than the sigma stage, especially in difficult negative regimes.

---

## Mathematical Ideas Used

### 1. Log Return

For asset price `S_t`:

`r_t = log(S_t / S_(t-1))`

### 2. Excess Return

If `r_f` is the daily risk-free rate:

`r_t^excess = r_t - r_f`

### 3. Volatility Target

Future volatility is estimated using rolling / forward windows of excess returns.

### 4. Drift–Diffusion Intuition

The project is inspired by stochastic differential equations of the form:

`dS_t = mu(t, S_t) dt + sigma(t, S_t) dW_t`

where:
- `mu` is drift
- `sigma` is diffusion / volatility

### 5. Simplified Risk Premium Relation

The project uses a simplified common-lambda relation:

`mu_excess(i,t) ≈ sigma(i,t) × lambda(t)`

where:
- `sigma(i,t)` is asset-level volatility
- `lambda(t)` is a common scalar market-premium signal

---

## Key Results

### Sigma / Covariance Stage
This is the strongest stage of the project.

The learned covariance-driven sigma:
- tracks the target volatility regime reasonably well
- captures major volatility spikes
- outperforms simple rolling and EWMA baselines on held-out test data

### Lambda Stage
The learned lambda:
- varies through time
- captures broad risk-premium regimes
- is smoother than the realized future Sharpe target
- should be interpreted as a common market-premium signal, not a direct Sharpe clone

### Mu Stage
The mu stage:
- shows some regime awareness
- is useful as a constructed excess-drift signal
- still underreacts in several negative future regimes

---

## Main Final Plots

The final report uses these key plots:

- **Covariance components plot**  
  Shows factor sigma, idiosyncratic sigma, raw marginal sigma, final marginal sigma, and target sigma.

- **Lambda vs future market Sharpe plot**  
  Shows how the learned lambda behaves relative to a noisy future market Sharpe target.

- **Lambda training vs validation objective plot**  
  Shows that the model begins mild overfitting after the best validation epoch, and that early stopping is used.

- **Held-out mu cross-sectional mean plot**  
  Shows the final downstream behavior of predicted mean excess drift and where negative regimes are still missed.

---

## What the Project Achieves

This project successfully builds a full multi-stage pipeline that:

- processes multi-asset financial data
- estimates conditional volatility structure
- learns a common time-varying market risk-premium signal
- constructs a simplified excess-drift estimate
- evaluates performance out of sample using chronological splits

The project is strongest in:
- **volatility / covariance estimation**
- **risk-premium regime interpretation**

The project is weaker in:
- **exact return prediction**
- **fully general structural market-price-of-risk estimation**

---

## Limitations

- This is a **discretized Neural SDE–inspired framework**, not a full exact continuous-time implementation.
- Lambda is modeled as a **common scalar signal shared across assets**.
- The one-factor/common-lambda formulation is a simplification.
- Mu is constructed approximately from sigma and lambda.
- The pipeline is stronger on volatility modeling than on exact excess-return forecasting.

---

## Future Work

Possible future improvements include:

- extending from a common scalar lambda to richer multi-factor formulations
- improving downside sensitivity in mu estimation
- testing on broader universes beyond NIFTY 50
- comparing against stronger market-premium baselines
- exploring more faithful continuous-time Neural SDE implementations

---

## How to Run

Run the pipeline step by step from the repository root:

```bash
python step0_fetch_nifty50_panel.py
python step0_fetch_risk_free.py
python step0_resolve_nifty50_panel.py
python step1_preprocess_panel.py
python step2_train_covariance.py
python step3_evaluate_covariance.py
python step4_fit_exposures.py
python step5_train_lambda.py
python step6_export_market_params.py
