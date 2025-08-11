import re
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# config 
PRICE_RE = re.compile(r"\$[\s]*([0-9][0-9,]*)")
RENAME_MAP = {
    "post_id": "id",
    "full_text": "desc",
    "price_text": "price_txt",
    "attrs_text": "attrs",
    "hood_text": "hood",
    "posted_text": "post_date",  # keep + rename
}
NUM_COLS  = ["price","beds","baths","sqft","lat","lon"]
BOOL_COLS = ["furnished","pets_allowed","utilities_included","parking_available"]

# Anchors
UBC      = (49.2606, -123.2460)
SFU_BBY  = (49.2775, -122.9146)
DOWNTOWN = (49.2859, -123.1207)
TRANSIT = np.array([
    [49.2856, -123.1110], [49.2833, -123.1165], [49.2796, -123.1098],
    [49.2626, -123.0693], [49.2272, -123.0026], [49.2664, -123.0017],
    [49.2484, -122.8970], [49.1896, -122.8489], [49.1664, -123.1365],
], dtype=float)

FINAL_COLS = [
    "id","title","price","beds","baths","sqft",
    "furnished","utilities_included","parking_available","pets_allowed",
    "city","lat","lon","post_date","student_flag",
    "d_ubc_km","d_sfu_km","d_dt_km","d_transit_km","near_transit",
    "url"
]

# helpers
def to_num(s): return pd.to_numeric(s, errors="coerce")

def parse_price_from_text(txt: str):
    if not isinstance(txt, str): return np.nan
    m = PRICE_RE.search(txt)
    if not m: return np.nan
    try: return float(m.group(1).replace(",", ""))
    except: return np.nan

def haversine_vec(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2*R*np.arcsin(np.sqrt(a))

def min_distance_to_transit(lat, lon):
    lat = np.asarray(lat, dtype=float); lon = np.asarray(lon, dtype=float)
    mask = ~np.isnan(lat) & ~np.isnan(lon)
    out = np.full(lat.shape, np.nan, dtype=float)
    if not mask.any(): return out
    l1 = np.repeat(lat[mask][:,None], TRANSIT.shape[0], axis=1)
    g1 = np.repeat(lon[mask][:,None], TRANSIT.shape[0], axis=1)
    l2 = np.repeat(TRANSIT[:,0][None,:], l1.shape[0], axis=0)
    g2 = np.repeat(TRANSIT[:,1][None,:], g1.shape[0], axis=0)
    d = haversine_vec(l1, g1, l2, g2)
    out[mask] = d.min(axis=1)
    return out

def nonempty(df, col):
    if col not in df.columns: return pd.Series(False, index=df.index)
    s = df[col]
    return s.notna() & (s.astype(str).str.strip() != "")

def coalesce_duplicate_column(df, name):
    """If multiple columns share `name`, bfill across them row-wise and keep one."""
    mask = (df.columns == name)
    if mask.sum() <= 1:
        return df
    sub = df.loc[:, mask]                    # all dup columns
    coalesced = sub.bfill(axis=1).iloc[:, 0] # first non-null across dups
    # drop all dups, then add single column
    df = df.loc[:, ~mask].copy()
    df[name] = coalesced
    return df

# main 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile",  default="data/rentals_raw.csv")
    ap.add_argument("--outfile", default="data/rentals_clean.csv")
    ap.add_argument("--min_price", type=float, default=600)
    ap.add_argument("--max_price", type=float, default=2500)
    ap.add_argument("--max_age_days", type=int, default=60, help="Only keep posts in last N days")
    ap.add_argument("--fast", action="store_true", help="Skip distance/transit features (faster)")
    args = ap.parse_args()

    infile = Path(args.infile); outfile = Path(args.outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    if not infile.exists():
        raise SystemExit(f"[ERR] raw file not found: {infile}")

    df = pd.read_csv(infile, low_memory=False)
    n_raw = len(df)

    # rename (includes posted_text -> post_date)
    df = df.rename(columns={k:v for k,v in RENAME_MAP.items() if k in df.columns})

    # coalesce duplicate 'post_date' if both existed
    if (df.columns == "post_date").sum() > 1:
        df = coalesce_duplicate_column(df, "post_date")

    # strip text ONLY for object dtype (avoid .str on datetimes/numerics)
    for c in ["title","city","url","desc","price_txt","post_date"]:
        if c in df.columns and pd.api.types.is_object_dtype(df[c]):
            df[c] = df[c].astype(str).str.strip()

    # parse post_date (tz-aware) and filter by recency
    if "post_date" in df.columns:
        # parse to tz-aware UTC, then convert to America/Vancouver and compare with tz-aware cutoff
        df["post_date"] = pd.to_datetime(df["post_date"], errors="coerce", utc=True)
        df["post_date"] = df["post_date"].dt.tz_convert("America/Vancouver")
        if args.max_age_days and args.max_age_days > 0:
            cutoff = pd.Timestamp.now(tz="America/Vancouver").normalize() - pd.Timedelta(days=args.max_age_days)
            df = df[df["post_date"] >= cutoff].copy()
        # store as plain date in output
        df["post_date"] = df["post_date"].dt.tz_localize(None).dt.date

    # build price (use price_txt/title/desc before dropping them)
    if "price" not in df.columns: df["price"] = np.nan
    df["price"] = to_num(df["price"])
    for src in ["price_txt","title","desc"]:
        if src in df.columns:
            m = df["price"].isna()
            if m.any():
                df.loc[m,"price"] = df.loc[m,src].apply(parse_price_from_text)

    # coerce numerics/booleans
    for c in NUM_COLS:
        if c in df.columns: df[c] = to_num(df[c])
    for c in BOOL_COLS:
        if c in df.columns:
            if pd.api.types.is_object_dtype(df[c]):
                df[c] = (df[c].astype(str).str.strip().str.lower()
                         .replace({"true": True, "false": False, "nan": np.nan, "none": np.nan, "": np.nan}))
            df[c] = df[c].astype("boolean")

    # dedupe by url
    if "url" in df.columns:
        df = df.sort_values("url").drop_duplicates("url", keep="last")

    # price band 600–2500
    df = df[df["price"].between(args.min_price, args.max_price, inclusive="both")].copy()

    # require price + title (city optional)
    df = df[ nonempty(df,"price") & nonempty(df,"title") ].copy()

    # drop raw-only text cols now
    drop_now = [c for c in ["raw_html_snippet","attrs","hood","desc","price_txt"] if c in df.columns]
    if drop_now: df = df.drop(columns=drop_now)

    # student keyword flag
    df["student_flag"] = df["title"].astype(str).str.contains(
        r"\b(?:student|sfu|ubc|langara|douglas|bcit|college|university)\b", case=False, regex=True
    )

    # distances unless --fast (compute only where coords exist)
    if not args.fast and {"lat","lon"}.issubset(df.columns):
        lat = df["lat"].to_numpy(dtype=float)
        lon = df["lon"].to_numpy(dtype=float)
        have = ~np.isnan(lat) & ~np.isnan(lon)
        if have.any():
            df.loc[have, "d_ubc_km"] = haversine_vec(lat[have], lon[have], UBC[0], UBC[1])
            df.loc[have, "d_sfu_km"] = haversine_vec(lat[have], lon[have], SFU_BBY[0], SFU_BBY[1])
            df.loc[have, "d_dt_km"]  = haversine_vec(lat[have], lon[have], DOWNTOWN[0], DOWNTOWN[1])
            df.loc[have, "d_transit_km"] = min_distance_to_transit(lat[have], lon[have])
            df["near_transit"] = df["d_transit_km"] <= 0.8

    # final columns only
    keep_cols = [c for c in FINAL_COLS if c in df.columns]
    df = df[keep_cols]

    # save
    df.to_csv(outfile, index=False)
    print(f"[CLEAN] raw rows: {n_raw}")
    print(f"[CLEAN] final rows (price {args.min_price}-{args.max_price}, <= {args.max_age_days} days): {len(df)} -> {outfile}")
    if len(df):
        print(df.head(5).to_string())

if __name__ == "__main__":
    main()
