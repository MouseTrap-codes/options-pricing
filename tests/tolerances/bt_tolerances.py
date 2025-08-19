BT_TOL_EURO = {"rel": 1e-6, "abs": 1e-4}
BT_EURO_TOL_OVERRIDES = {
    "High_vol_european_call": {"rel": 2e-6, "abs": 5e-3},
    "Very_high_volatility_european": {
        "rel": 2e-6,
        "abs": 3e-1,
    },  # if you include this one
    "Future_european_call": {"rel": 2e-6, "abs": 2e-3},
    "Future_european_put": {"rel": 2e-6, "abs": 2e-3},
    "Deep_ITM_european_call": {"rel": 2e-6, "abs": 2e-3},
    "Min_time_steps_european": {"rel": 2e-6, "abs": 2e-3},
}

BT_EURO_TOL_OVERRIDES.update(
    {
        "ATM_european_call": {"rel": 2e-6, "abs": 4e-4},
        "ATM_european_put": {"rel": 2e-6, "abs": 4e-4},
        "Deep_ITM_european_put": {"rel": 2e-6, "abs": 4e-4},
        "High_dividend_european_put": {"rel": 2e-6, "abs": 5e-4},
        "Very_low_vol_european_put": {"rel": 2e-6, "abs": 4e-4},
        "Long_term_OTM_european_put": {"rel": 2e-6, "abs": 4e-4},
        "Zero_interest_rate_european": {"rel": 2e-6, "abs": 4e-4},
    }
)

BT_AMER_TOL_OVERRIDES = {
    "ATM_american_call": {"rel": 2e-6, "abs": 5e-4},
    "Future_american_put": {"rel": 2e-6, "abs": 2e-3},
    "High_vol_american_call": {"rel": 2e-6, "abs": 4e-3},
    "Deep_ITM_american_call": {"rel": 2e-6, "abs": 2e-3},
    "ATM_american_put": {"rel": 2e-6, "abs": 5e-4},
    "Future_american_call": {"rel": 2e-6, "abs": 2e-3},
    "High_dividend_american_put": {"rel": 2e-6, "abs": 5e-4},
    "Long_term_OTM_american_put": {"rel": 2e-6, "abs": 5e-4},
    "Min_time_steps_american": {"rel": 2e-6, "abs": 2e-3},
    "Zero_interest_rate_american": {"rel": 2e-6, "abs": 5e-4},
    "Very_high_volatility_american": {
        "rel": 2e-6,
        "abs": 5e-1,
    },  # larger due to extreme vol
}
