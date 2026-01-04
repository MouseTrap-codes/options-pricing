# Options Pricing Platform

Web-based options pricing toolkit with Black-Scholes and Binomial Tree models. Supports European and American exercise styles across multiple asset types (stocks, currencies, futures). Includes live market data integration, implied volatility solver, and P&L visualization.

## Live Demo

**Try it here:** https://options-pricing-analytics-platform.streamlit.app/

## What It Does

**Interactive Web App:** Price options using Black-Scholes or Binomial Tree models with live market data from Yahoo Finance

**Performance Benchmarks:** Compares naive recursive implementation against JAX-optimized dynamic programming approach

**Comprehensive Testing:** 50+ test cases validated against QuantLib across different market conditions and edge cases

## Quick Start
```bash
# install dependencies
poetry install

# run web app locally
poetry run streamlit run Options_Analytics_Platform.py

# run tests
poetry run pytest

# generate benchmark data
poetry run python benchmarks/performance.py
```

## Project Structure
```
models/
├── black_scholes.py           # BSM pricing + Greeks
├── naive_binomial_tree.py     # Recursive binary tree implementation
├── optimized_binomial_tree.py # JAX-compiled DP implementation
└── binomial_tree_utils.py     # Shared tree utilities (u, d, p calculations)

benchmarks/
├── performance.py             # Speed & memory benchmarking
├── benchmark_utils.py         # Timing & profiling helpers
├── make_plots.py              # Plotly visualizations
└── results.csv                # Raw benchmark data

test_case_generation/
├── bsm_test_cases_generator.py       # QuantLib BSM reference prices
├── bt_euro_test_cases_generator.py   # QuantLib European binomial
└── bt_amer_test_cases_generator.py   # QuantLib American binomial

tests/
├── test_black_scholes.py
├── test_optimized_binomial_tree.py
└── tolerances/                # Per-metric tolerance configs

pages/
└── 1_Performance_Benchmarks.py  # Streamlit benchmark dashboard

option_chain_helpers.py        # Live data fetching & IV solver
pnl_helpers.py                 # P&L computation & Plotly charts
```

## Design Decisions

**JAX + Dynamic Programming for Binomial Trees:** The naive recursive implementation hits exponential time complexity O(2^N). DP reduces this to O(N²), and JAX's XLA compilation parallelizes array operations. Result: ~135,000x speedup at N=10,000 timesteps. Enables real-time pricing with high-resolution trees.

**Two Binomial Implementations:** Kept the recursive version for educational clarity (shows the actual tree structure) and built the optimized version for production use. Both produce identical results - tests verify this.

**QuantLib for Test Case Generation:** Can't trust self-generated test cases when building a pricer. Used QuantLib (industry-standard C++ library) to generate reference prices across 50+ scenarios. Wrote custom generators that handle all the asset-type nuances (dividend yields, foreign rates, futures).

**Streamlit for UI:** Needed something that could handle live plots, real-time market data, and complex input forms without writing JavaScript. Streamlit's caching handles the yfinance API calls efficiently.

**Implied Volatility Solver:** Market doesn't quote volatility directly - it quotes option prices. Built Brent's method solver that inverts the pricing model to back out σ from market premiums. Handles edge cases like OTM options with zero bid/ask spread.

**Live Market Data Integration:** yfinance provides free options chain data. Implemented strike-matching logic with tolerance checks since exchange strikes don't always align perfectly with user input. Caches results for 5 minutes to avoid rate limits.

**Error Collection Strategy:** For testing, collect all assertion failures instead of stopping at first failure. Helps identify systematic issues (e.g., "all high-volatility cases are failing" vs random noise).

## Models Implemented

### Black-Scholes-Merton

Closed-form solution for European options on dividend-paying stocks:

$$C = S_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2)$$
$$P = K e^{-rT} N(-d_2) - S_0 e^{-qT} N(-d_1)$$

where:
$$d_1 = \frac{\ln(S_0/K) + (r - q + \sigma^2/2)T}{\sigma\sqrt{T}}$$
$$d_2 = d_1 - \sigma\sqrt{T}$$

**Greeks calculated:**
- **Delta (Δ):** Sensitivity to underlying price changes
- **Gamma (Γ):** Sensitivity of Delta to underlying price
- **Vega (ν):** Sensitivity to volatility (per 1% change)
- **Theta (Θ):** Time decay (per day)
- **Rho (ρ):** Sensitivity to interest rates (per 1% change)

**Supported assets:** Non-dividend stocks, dividend-paying stocks

**Exercise style:** European only

---

### Binomial Tree (Cox-Ross-Rubinstein)

Discrete-time lattice model that simulates possible price paths:

1. Divide time to maturity into N steps: Δt = T/N
2. Calculate up/down factors: u = e^(σ√Δt), d = 1/u
3. Compute risk-neutral probability: p = (e^(bΔt) - d)/(u - d)
4. Build price tree forward, then value option backward

**Drift parameter b varies by asset type:**
- Non-dividend stock: b = r
- Dividend-paying stock: b = r - q
- Currency: b = r - r_f (domestic rate minus foreign rate)
- Futures: b = 0 (no drift under risk-neutral measure)

**Two implementations:**

**Naive (Recursive):**
- Binary tree data structure
- Post-order traversal for backward induction
- Time: O(2^N), Space: O(2^N)
- Useful for teaching/visualization

**Optimized (DP + JAX):**
- Single array stores each time layer
- JAX's `fori_loop` compiles to XLA
- Time: O(N²), Space: O(N)
- Production-ready for N ≤ 10,000

**Supported assets:** Stocks, dividend stocks, currencies, futures

**Exercise styles:** European, American

---

### Implied Volatility Solver

Given market price P_market, solve for σ such that:

$$\text{Model}(S_0, K, T, r, \sigma, ...) = P_{\text{market}}$$

Uses Brent's method (bracket-based root finding) with adaptive bounds:
- Start with σ ∈ [0.0001, 5.0]
- If model price > market price at upper bound → expand upward
- If model price < market price at lower bound → contract downward
- Converges to 10 decimal places in ~10-20 iterations

**Handles edge cases:**
- Deep OTM options (near-zero prices)
- High-volatility regimes (σ > 100%)
- Binomial trees with drift constraints (σ_min = 1.05|b|√Δt to ensure valid probabilities)

## Performance Results

Benchmarked on: American call, S=42, K=40, T=0.25, r=0.10, σ=0.20

### Naive vs Optimized (N ≤ 25)

| N  | Naive Time | Optimized Time | Speedup  | Naive Memory | Optimized Memory |
|----|------------|----------------|----------|--------------|------------------|
| 5  | 24.1 µs    | 192.3 µs       | 0.13x    | 0.01 MiB     | 0.65 MiB         |
| 10 | 748.3 µs   | 193.6 µs       | 3.9x     | 0.18 MiB     | 0.65 MiB         |
| 15 | 23,847 µs  | 194.7 µs       | 122x     | 5.63 MiB     | 0.65 MiB         |
| 20 | 762,184 µs | 195.2 µs       | 3,900x   | 180 MiB      | 0.65 MiB         |
| 25 | 24.4M µs   | 196.1 µs       | 124,000x | 5,760 MiB    | 0.65 MiB         |

### Optimized Performance (N ≤ 10,000)

| N     | Time     | Memory  |
|-------|----------|---------|
| 100   | 205 µs   | 0.66 MiB|
| 500   | 286 µs   | 0.68 MiB|
| 1,000 | 409 µs   | 0.71 MiB|
| 5,000 | 1,821 µs | 1.12 MiB|
| 10,000| 5,241 µs | 1.88 MiB|

**Key insight:** JAX overhead dominates for N < 10, but scales logarithmically while naive scales exponentially. Memory usage stays flat due to in-place array updates.

## Web App Features

### Manual Mode
- Input all parameters manually (S₀, K, T, r, σ, q/r_f)
- Select model (Black-Scholes or Binomial Tree)
- Choose option type (call/put), exercise style, asset type
- View price, Greeks (BSM only), and P&L chart

### Live Market Data Mode
- Enter ticker symbol (e.g., AAPL)
- Fetches current spot price and option chain from Yahoo Finance
- Select expiration date and strike from actual market quotes
- Computes implied volatility from market mid/last price
- Overlay model P&L vs market P&L for comparison
- Position Greeks scaling (contracts × multiplier)

### P&L Visualization
- Interactive Plotly chart: spot price vs profit/loss at expiration
- Breakeven markers
- Max gain/loss annotations (when finite)
- Shaded profit/loss regions
- Summary statistics table

## Testing Strategy

### Test Case Generation
- **QuantLib Reference Prices:** Generate expected outputs using industry-standard library
- **50+ Scenarios:** ATM, ITM, OTM, high vol, low vol, dividend, currency, futures, edge cases
- **JSON Format:** Stored in `tests/data/` for version control and reproducibility

### Tolerance Handling
- **Per-metric tolerances:** Theta requires looser tolerance (1e-7 rel) than Price (1e-8 rel)
- **Per-case overrides:** High-volatility cases get relaxed tolerances due to numerical instability
- **Binomial convergence:** Accepts ±0.01% error since trees approximate continuous models

### Validation Approach
1. Black-Scholes vs QuantLib BSM (exact match expected)
2. Binomial European vs QuantLib CRR tree (numerical convergence)
3. Binomial American vs QuantLib CRR tree (early exercise logic)
4. Naive vs Optimized implementations (exact match required)
5. Mathematical properties (put-call parity, delta relationships)

## Built With

- Python 3.13
- JAX 0.7 (XLA compilation, automatic differentiation)
- NumPy / SciPy (numerical routines)
- Streamlit (web framework)
- Plotly (interactive charts)
- yfinance (live market data)
- QuantLib (test case generation only - not a runtime dependency)
- Poetry (dependency management)
- pytest (testing framework)
- Ruff, Black, isort, mypy (code quality)

## References

Based on:
- *Options, Futures, and Other Derivatives* by John C. Hull (9th Edition)
- *Derivatives Markets* by Robert L. McDonald (3rd Edition)
- Cox, J.C., Ross, S.A., & Rubinstein, M. (1979). "Option pricing: A simplified approach." *Journal of Financial Economics*
