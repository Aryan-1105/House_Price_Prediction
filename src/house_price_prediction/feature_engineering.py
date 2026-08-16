import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the feature engineering used by the final training notebook."""

    data = df.copy()

    data["total_rooms"] = data["bedrooms"] + data["bathrooms"]

    data["area_per_bedroom"] = data["area"] / data["bedrooms"].replace(0, np.nan)

    data["area_per_bathroom"] = data["area"] / data["bathrooms"].replace(0, np.nan)

    data["amenity_count"] = (
        (data["mainroad"] == "yes").astype(int)
        + (data["guestroom"] == "yes").astype(int)
        + (data["basement"] == "yes").astype(int)
        + (data["hotwaterheating"] == "yes").astype(int)
        + (data["airconditioning"] == "yes").astype(int)
        + (data["prefarea"] == "yes").astype(int)
    )

    return data
