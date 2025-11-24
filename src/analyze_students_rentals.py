import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

IN = "data/rentals_clean.csv"
OUT = "figs"
SUMMARY_FILE = os.path.join(OUT, "rental_summary_combined.csv")
RF_IMP_FILE = os.path.join(OUT, "random_forest_importances.png")
RF_PRED_FILE = os.path.join(OUT, "rf_predictions_sample.csv")

plt.rcParams.update({"figure.figsize": (10, 7), "axes.grid": True, "figure.dpi": 140})
PALETTE = {"SFU": "#1f77b4", "UBC": "#2ca02c", "Downtown": "#d62728"}

# anchors (lat, lon)
UBC      = (49.2606, -123.2460)
SFU_BBY  = (49.2775, -122.9146)
DOWNTOWN = (49.2859, -123.1207)

def ensure_dir(d): os.makedirs(d, exist_ok=True)

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    la1 = np.radians(lat1); lo1 = np.radians(lon1)
    la2 = np.radians(lat2); lo2 = np.radians(lon2)
    dlat = la2 - la1; dlon = lo2 - lo1
    a = np.sin(dlat/2)**2 + np.cos(la1)*np.cos(la2)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def derive_nearest_area(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure d_ubc_km/d_sfu_km/d_dt_km, nearest_area, distance_to_nearest_km exist."""
    df = df.copy()
    have_latlon = {"lat","lon"}.issubset(df.columns) and df["lat"].notna().any() and df["lon"].notna().any()

    # Fill missing distance-to-anchors from lat/lon if possible
    if have_latlon:
        lat = df["lat"].to_numpy(dtype=float)
        lon = df["lon"].to_numpy(dtype=float)
        mask = ~np.isnan(lat) & ~np.isnan(lon)

        for (col, anchor) in [("d_ubc_km", UBC), ("d_sfu_km", SFU_BBY), ("d_dt_km", DOWNTOWN)]:
            if col not in df.columns:
                df[col] = np.nan
            need = df[col].isna() & mask
            if need.any():
                df.loc[need, col] = haversine_km(lat[need], lon[need], anchor[0], anchor[1])

    # Build nearest_area + distance_to_nearest_km if we have the 3 anchor distances
    if all(c in df.columns for c in ["d_ubc_km","d_sfu_km","d_dt_km"]):
        dist_mat = np.vstack([
            df["d_ubc_km"].to_numpy(dtype=float),
            df["d_sfu_km"].to_numpy(dtype=float),
            df["d_dt_km"].to_numpy(dtype=float)
        ]).T  # (n, 3)
        labels = np.array(["UBC","SFU","Downtown"])
        all_nan = np.isnan(dist_mat).all(axis=1)
        safe = np.where(np.isnan(dist_mat), np.inf, dist_mat)
        nearest_idx = np.argmin(safe, axis=1)
        df["nearest_area"] = np.where(all_nan, np.nan, labels[nearest_idx])
        df["distance_to_nearest_km"] = np.where(all_nan, np.nan, np.nanmin(dist_mat, axis=1))
    else:
        # fallback columns so later code doesn't crash
        if "nearest_area" not in df.columns: df["nearest_area"] = np.nan
        if "distance_to_nearest_km" not in df.columns: df["distance_to_nearest_km"] = np.nan
    return df

def histogram(series, bins, title, xlabel, fname, xlim=None):
    ensure_dir(OUT)
    fig, ax = plt.subplots()
    s = series.dropna()
    if len(s) == 0: plt.close(fig); return
    ax.hist(s, bins=bins, alpha=0.75)
    if xlim: ax.set_xlim(*xlim)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("Count")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, fname)); plt.close(fig)

def scatter(x, y, groups, xlabel, ylabel, title, fname):
    ensure_dir(OUT)
    fig, ax = plt.subplots()
    valid = (~pd.isna(x)) & (~pd.isna(y))
    x = x[valid]; y = y[valid]
    groups = groups[valid] if groups is not None else None

    if groups is None or groups.dropna().empty:
        ax.scatter(x, y, s=35, alpha=0.7)
    else:
        for lbl in sorted(pd.Series(groups.dropna().unique()).tolist()):
            m = (groups == lbl)
            ax.scatter(x[m], y[m], s=35, alpha=0.7, label=str(lbl), c=PALETTE.get(lbl, None))
        ax.legend(title="Nearest area", loc="best")

    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, fname)); plt.close(fig)

def quartile_agg(s: pd.Series) -> pd.Series:
    s = s.dropna()
    if len(s) == 0:
        return pd.Series({"count":0,"median":np.nan,"p25":np.nan,"p75":np.nan})
    return pd.Series({
        "count": int(len(s)),
        "median": float(np.median(s)),
        "p25": float(np.percentile(s, 25)),
        "p75": float(np.percentile(s, 75)),
    })

def summary_block(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Return summary (count/median/p25/p75) for group_col. If missing, return empty DF."""
    if group_col not in df.columns:
        return pd.DataFrame(columns=["summary_type","group","count","median","p25","p75"])
    tmp = df.groupby(group_col, dropna=False)["price"].apply(quartile_agg).reset_index()
    tmp.rename(columns={group_col: "group"}, inplace=True)
    tmp.insert(0, "summary_type", group_col)
    return tmp

def train_random_forest(df: pd.DataFrame):
    """Train RF on current df and write feature importances + sample predictions."""
    # target
    y = df["price"].astype(float)

    # numeric features (use available ones)
    num_feats = [c for c in [
        "sqft","lat","lon","distance_to_nearest_km","d_ubc_km","d_sfu_km","d_dt_km"
    ] if c in df.columns]

    # boolean features -> 0/1
    bool_feats = [c for c in [
        "furnished","utilities_included","near_transit","parking_available","pets_allowed","student_flag"
    ] if c in df.columns]

    # categorical
    cat_feats = [c for c in ["nearest_area"] if c in df.columns]

    if len(num_feats) + len(bool_feats) + len(cat_feats) == 0:
        print("[RF] Not enough features to train. Skipping.")
        return

    X = df[num_feats + bool_feats + cat_feats].copy()

    # booleans to 0/1 (floats so we can fill NaN)
    for c in bool_feats:
        X[c] = X[c].astype("Int64").astype(float)

    # fill numeric NaNs with medians
    for c in num_feats + bool_feats:
        if c in X.columns:
            X[c] = X[c].fillna(X[c].median())

    # one-hot encode categorical
    if len(cat_feats):
        X = pd.get_dummies(X, columns=cat_feats, dummy_na=True)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # model
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # eval
    y_pred = rf.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    print(f"[RF] MAE: {mae:.1f}   R²: {r2:.3f}   (n_test={len(y_test)})")

    # feature importances
    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1]
    top_k = min(12, len(idx))
    top_idx = idx[:top_k]
    plt.figure(figsize=(9, 6))
    plt.barh(range(top_k), importances[top_idx][::-1])
    plt.yticks(range(top_k), [X.columns[i] for i in top_idx][::-1])
    plt.xlabel("Importance")
    plt.title("Random Forest Feature Importances")
    plt.tight_layout()
    ensure_dir(OUT)
    plt.savefig(RF_IMP_FILE)
    plt.close()
    print(f"[SAVE] {RF_IMP_FILE}")

    # predictions sample with context
    sample = X_test.copy()
    sample["price_actual"] = y_test
    sample["price_pred"]   = y_pred
    sample["abs_error"]    = (sample["price_pred"] - sample["price_actual"]).abs()

    # attach human columns if available, but avoid name collisions
    attach_cols = [c for c in ["title","url","nearest_area","city","sqft","furnished","utilities_included","near_transit"]
                   if c in df.columns]
    non_overlap = [c for c in attach_cols if c not in sample.columns]
    if non_overlap:
        sample = sample.join(df.loc[sample.index, non_overlap])

    # nicer column order
    front = [c for c in ["title","url","nearest_area","city","sqft","furnished","utilities_included","near_transit"]
             if c in sample.columns]
    front += ["price_actual","price_pred","abs_error"]
    ordered = front + [c for c in sample.columns if c not in front]
    sample[ordered].sort_values("abs_error", ascending=False).head(50).to_csv(RF_PRED_FILE, index=False)
    print(f"[SAVE] {RF_PRED_FILE}")

def main():
    if not os.path.exists(IN):
        raise SystemExit(f"Missing {IN}. Run clean_rentals.py first.")
    ensure_dir(OUT)

    df = pd.read_csv(IN)
    if "price" not in df.columns:
        raise SystemExit("clean file has no 'price' column.")

    # Student budget cap for analysis/ML
    df = df[df["price"].between(0, 2500, inclusive="both")].copy()

    # derive nearest_area & distances if needed
    df = derive_nearest_area(df)

    # ----- combined summary (one CSV) -----
    blocks = []
    blocks.append(summary_block(df, "nearest_area"))
    for col in ["utilities_included","furnished","near_transit","pets_allowed","parking_available"]:
        blocks.append(summary_block(df, col))
    combined = pd.concat([b for b in blocks if not b.empty], ignore_index=True)
    combined.to_csv(SUMMARY_FILE, index=False)
    print(f"[SAVE] {SUMMARY_FILE}")

    # ----- cheapest & most expensive area (by median) -----
    if "nearest_area" in df.columns and df["nearest_area"].notna().any():
        area_stats = df.groupby("nearest_area")["price"].median().dropna().sort_values()
        print("\n=== Area Price Summary (Median) ===")
        for area, price in area_stats.items():
            print(f"{area}: ${price:.0f}")
        print(f"\nCheapest area overall: {area_stats.index[0]} (${area_stats.iloc[0]:.0f})")
        print(f"Most expensive area overall: {area_stats.index[-1]} (${area_stats.iloc[-1]:.0f})")
    else:
        print("\n[WARN] nearest_area not available (or all NaN). Skipping area ranking.")

    # ===== Random Forest regression (predict price) =====
    if len(df) >= 50:
        train_random_forest(df)
    else:
        print("[RF] Not enough rows to train a model. Skipping RF part.")

    # ----- plots -----
    histogram(
        df["price"], bins=25,
        title="Distribution of Monthly Rent (≤ $2500)", xlabel="Rent (CAD)",
        fname="hist_price.png", xlim=(0, 2500)
    )

    if "distance_to_nearest_km" in df.columns and df["distance_to_nearest_km"].notna().any():
        scatter(
            df["distance_to_nearest_km"], df["price"], df.get("nearest_area"),
            xlabel="Distance to Nearest Major Area (km)", ylabel="Rent (CAD)",
            title="Rent vs Distance to Nearest Major Area",
            fname="scatter_price_vs_distance.png"
        )

    if "sqft" in df.columns and df["sqft"].notna().sum() > 0:
        s95 = float(df["sqft"].quantile(0.95))
        m = df["sqft"] <= s95
        scatter(
            df.loc[m, "sqft"], df.loc[m, "price"], df.loc[m].get("nearest_area"),
            xlabel="Square feet (<= 95th pct)", ylabel="Rent (CAD)",
            title="Rent vs Size", fname="scatter_price_vs_sqft.png"
        )

    if {"lat","lon"}.issubset(df.columns) and df["lat"].notna().any() and df["lon"].notna().any():
        scatter(
            df["lon"], df["lat"], df.get("nearest_area"),
            xlabel="Longitude", ylabel="Latitude",
            title="Map: Listings by Nearest Area",
            fname="scatter_geo.png"
        )

    print("[DONE] figures + CSVs saved in ./figs")

if __name__ == "__main__":
    main()
