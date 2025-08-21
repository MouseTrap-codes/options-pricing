import pandas as pd
import pandera.pandas as pa


def create_schema() -> pa.DataFrameSchema:
    schema = pa.DataFrameSchema(
        {
            "Time Steps (N)": pa.Column(pa.Float64),
            "Naive (Recursive) Time (µs)": pa.Column(pa.Float64, nullable=True),
            "Naive (Recursive) Peak Memory (MiB)": pa.Column(pa.Float64, nullable=True),
            "Optimized (DP + JAX) Time (µs)": pa.Column(pa.Float64),
            "Optimized (DP + JAX) Peak Memory (MiB)": pa.Column(pa.Float64),
        },
        strict=True,
        coerce=True,
    )

    return schema


def create_df(file: str) -> pd.DataFrame:
    schema = create_schema()

    try:
        df = pd.read_csv(file, na_values=["NA"])
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{file}' not found")
    except pd.errors.EmptyDataError:
        raise ValueError("File is empty")

    if df.empty:
        raise ValueError("CSV file contains no data")

    df = schema.validate(df)

    return df
