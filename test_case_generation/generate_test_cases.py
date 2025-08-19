import argparse
import json

from .bsm_test_cases_generator import build_payload_bsm
from .bt_amer_test_cases_generator import build_payload_binomial_amer
from .bt_euro_test_cases_generator import build_payload_binomial_euro


def write_json(payload, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bt-euro-out",
        default="tests/data/bt_european_test_cases.json",
        help="Path for binomial European test cases JSON",
    )
    parser.add_argument(
        "--bt-amer-out",
        default="tests/data/bt_american_test_cases.json",
        help="Path for binomial American test cases JSON",
    )
    parser.add_argument(
        "--bsm-out",
        default="tests/data/bs_cases.json",
        help="Path for Black–Scholes test cases JSON",
    )
    args = parser.parse_args()

    # Binomial-European cases
    write_json(build_payload_binomial_euro(), args.bt_euro_out)

    # Binomial-American cases
    write_json(build_payload_binomial_amer(), args.bt_amer_out)

    # Black–Scholes cases
    write_json(build_payload_bsm(), args.bsm_out)


if __name__ == "__main__":
    main()
