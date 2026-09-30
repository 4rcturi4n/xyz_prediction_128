"""
Vendored copy of the x/y dataset-building utilities from
printer_ml/src/printer_ml/dataset_maker.py (only the parts this project
needs). Rebuilds video_xy_dataset.csv from raw video + excel folders --
run this fresh on whatever machine has the "Early prediction" data, since
absolute paths won't carry over between machines.
"""

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from openpyxl import load_workbook


@dataclass(frozen=True)
class FolderPair:
    videos_folder: str
    xl_folder: str
    label: str


FolderPairLike = Union[FolderPair, Tuple[str, str, str]]


def p_from_mp4(path: str) -> Optional[float]:
    """Extract power from mp4 filename. Ignores big numbers (>=500, likely velocity)."""
    base = os.path.basename(path)
    name_no_ext = os.path.splitext(base)[0]
    nums = re.findall(r"\d+[.,]?\d*", name_no_ext)
    for tok in nums:
        num_str = tok.replace(",", ".")
        try:
            val = float(num_str)
        except ValueError:
            continue
        if val < 500:
            return val
    print("WARNING: no valid power found in video name:", base)
    return None


def p_from_xlsx(name: str) -> Optional[float]:
    """Extract power from excel filename, e.g. '16-12_5,1000.xlsx' -> 12.5."""
    base = os.path.basename(name)
    name_no_ext = os.path.splitext(base)[0]
    parts = name_no_ext.split("-")
    if len(parts) < 2:
        print("WARNING: unexpected excel name format:", base)
        return None
    second = parts[1]
    if "," in second:
        num_str = second.split(",")[0].replace("_", ".")
    else:
        num_str = second.split("_")[0]
    try:
        return float(num_str)
    except ValueError:
        print("WARNING: bad power format in excel name:", base, "->", num_str)
        return None


def match_folder(
    videos: List[str],
    xl_names: List[str],
    xl_folder: str,
    label: str = "",
) -> Tuple[List[str], List[str], List[str]]:
    """
    Match video<->excel within the same folder by power.
    - video with no matching-power excel => dropped
    - excel with no matching-power video => dropped
    - duplicate excels at the same power => keep the last one
    """
    excel_powers_by_name = {}
    for xn in xl_names:
        p = p_from_xlsx(xn)
        if p is None:
            print(f"[{label}] Delete EXCEL '{os.path.join(xl_folder, xn)}' because cannot parse power")
        else:
            excel_powers_by_name[xn] = p

    power_to_excel_names = {}
    for name, p in excel_powers_by_name.items():
        power_to_excel_names.setdefault(p, []).append(name)

    excel_by_power = {}
    for p, names in power_to_excel_names.items():
        names_sorted = sorted(names, key=lambda n: xl_names.index(n))
        if len(names_sorted) > 1:
            for del_name in names_sorted[:-1]:
                print(f"[{label}] Duplicate power={p}: delete first EXCEL '{os.path.join(xl_folder, del_name)}'")
        excel_by_power[p] = names_sorted[-1]

    video_powers = [p_from_mp4(v) for v in videos]

    matched_videos, matched_xl_names, matched_xl_paths = [], [], []
    used_powers = set()

    for v, pv in zip(videos, video_powers):
        if pv is None:
            print(f"[{label}] Delete VIDEO '{v}' because cannot parse power")
            continue
        xl_name = excel_by_power.get(pv)
        if xl_name is None:
            print(f"[{label}] Delete VIDEO '{v}' (power={pv}) because no Excel data in same folder")
        else:
            matched_videos.append(v)
            matched_xl_names.append(xl_name)
            matched_xl_paths.append(os.path.join(xl_folder, xl_name))
            used_powers.add(pv)

    for p, name in excel_by_power.items():
        if p not in used_powers:
            print(f"[{label}] Delete EXCEL '{os.path.join(xl_folder, name)}' (power={p}) because no video data in same folder")

    print(f"[{label}] Kept {len(matched_videos)} matched pairs.")
    return matched_videos, matched_xl_names, matched_xl_paths


def read_mean_and_error(xl_path: str, sheet_name: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Reads G25 (mean of the 21 repeat measurements) and G26 (their std dev)
    directly from the given sheet -- the sheet's own precomputed values,
    not re-derived from the raw rows. No calibration factor applied.
    Error (G26) is optional; a missing G26 doesn't affect the mean.
    """
    try:
        wb = load_workbook(xl_path, data_only=True)
    except Exception:
        return None, None
    if sheet_name not in wb.sheetnames:
        return None, None
    ws = wb[sheet_name]

    mean_val = ws["G25"].value
    try:
        mean_val = float(mean_val)
    except (TypeError, ValueError):
        mean_val = None

    error_val = ws["G26"].value
    try:
        error_val = float(error_val)
    except (TypeError, ValueError):
        error_val = None

    return mean_val, error_val


def read_z_mean_and_error(xl_path: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Reads axial resolution (z) from the sheet's own "z" tab. Unlike x/y,
    z needs a 0.7 calibration factor -- confirmed by checking the sheet's
    own H25 cell against G25/0.7 across every workbook (exact match,
    0 mismatches out of 113 checked): H25 = G25 / 0.7, i.e. H is already
    the calibrated mean, so we read it directly.

    For the error, H26 is NOT the calibrated std dev -- it's a relative
    coefficient of variation expressed as a percentage (confirmed: H26 ==
    G26/G25*100, and I26 literally contains the unit label "%"). Using it
    as an absolute error would be wrong (wrong units, wrong scale). So the
    error is computed the same way the mean was calibrated: G26/0.7, which
    puts it in the same physical units as H25.

    Some workbooks have '#DIV/0!' in G25/H25 (division-by-zero in the
    sheet's own formula, e.g. from a std dev of zero across repeats) or no
    "z" sheet at all -- both return (None, None) and get dropped downstream,
    same as x/y's missing-data handling.
    """
    try:
        wb = load_workbook(xl_path, data_only=True)
    except Exception:
        return None, None
    if "z" not in wb.sheetnames:
        return None, None
    ws = wb["z"]

    mean_val = ws["H25"].value
    if not isinstance(mean_val, (int, float)):
        mean_val = None

    g26 = ws["G26"].value
    if not isinstance(g26, (int, float)):
        error_val = None
    else:
        error_val = g26 / 0.7

    return mean_val, error_val


def read_power_mw(xl_path: str) -> Optional[float]:
    """Reads A1 for power (e.g. '15,5 mW'). Same value on every sheet in these workbooks."""
    wb = load_workbook(xl_path, data_only=True)
    raw_a1 = str(wb.active["A1"].value)
    m = re.search(r"\d+[.,]?\d*", raw_a1)
    if not m:
        return None
    try:
        return float(m.group().replace(",", "."))
    except ValueError:
        return None


def build_xy_dataset(
    folder_pairs: List[FolderPairLike],
    out_csv: str = "data/processed/video_xy_dataset.csv",
) -> pd.DataFrame:
    """
    Match videos to excels by power within each folder pair, drop anything
    without a match on either side, read x_resolution/x_error/y_resolution/
    y_error, drop rows with missing/zero x or y (error is nice-to-have,
    not required).
    """
    video_list, xlsx_aligned_names, xl_aligned_paths = [], [], []

    for i, pair in enumerate(folder_pairs, start=1):
        if isinstance(pair, FolderPair):
            videos_folder, xl_folder, label = pair.videos_folder, pair.xl_folder, pair.label
        else:
            videos_folder, xl_folder, label = pair

        if not os.path.isdir(videos_folder):
            raise FileNotFoundError(f"Videos folder not found: {videos_folder}")
        if not os.path.isdir(xl_folder):
            raise FileNotFoundError(f"Excel folder not found: {xl_folder}")

        videos = [os.path.join(videos_folder, f) for f in os.listdir(videos_folder) if f.endswith(".mp4")]
        xls = [f for f in os.listdir(xl_folder) if f.endswith(".xlsx")]

        print(f"\nPAIR {i} [{label}] Found videos={len(videos)} excels={len(xls)}")

        v, xl_names, xl_paths = match_folder(videos, xls, xl_folder, label=label)
        video_list += v
        xlsx_aligned_names += xl_names
        xl_aligned_paths += xl_paths

    print("\nTOTAL matched videos:", len(video_list))
    print("TOTAL matched excels:", len(xl_aligned_paths))

    powers, xs, x_errs, ys, y_errs = [], [], [], [], []
    for path in xl_aligned_paths:
        powers.append(read_power_mw(path))
        x_mean, x_err = read_mean_and_error(path, "x")
        y_mean, y_err = read_mean_and_error(path, "y")
        xs.append(x_mean); x_errs.append(x_err)
        ys.append(y_mean); y_errs.append(y_err)

    df = pd.DataFrame({
        "video_name": [os.path.splitext(os.path.basename(v))[0] for v in video_list],
        "video_path": video_list,
        "xl_name": xlsx_aligned_names,
        "xl_path": xl_aligned_paths,
        "power_mW": powers,
        "x_resolution": xs,
        "x_error": x_errs,
        "y_resolution": ys,
        "y_error": y_errs,
    })

    def make_video_id(path):
        folder = os.path.basename(os.path.dirname(path))
        name = os.path.splitext(os.path.basename(path))[0]
        return f"{folder}\\{name}"

    df["video_id"] = df["video_path"].apply(make_video_id)

    for col in ["x_resolution", "x_error", "y_resolution", "y_error"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df[
        df["x_resolution"].notna() & (df["x_resolution"] != 0)
        & df["y_resolution"].notna() & (df["y_resolution"] != 0)
    ].copy()
    after = len(df)
    print(f"Dropped {before - after} rows with invalid x/y resolution (missing / 0). Kept {after} rows.")
    print(f"  (of those kept, {df['x_error'].isna().sum()} missing x_error, {df['y_error'].isna().sum()} missing y_error)")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("Saved dataset to:", out_csv)

    return df


def build_z_dataset(
    folder_pairs: List[FolderPairLike],
    out_csv: str = "data/processed/video_z_dataset.csv",
) -> pd.DataFrame:
    """
    Same video<->excel matching as build_xy_dataset (identical videos, same
    6 folder pairs) but reads z_resolution/z_error from the "z" sheet
    instead of x/y from the "x"/"y" sheets. The set of valid rows differs
    from x/y's: some workbooks have a usable x/y but a '#DIV/0!' z (or no
    z sheet at all), and vice versa -- so this is built and filtered
    independently rather than reusing video_xy_dataset.csv's row set.
    """
    video_list, xlsx_aligned_names, xl_aligned_paths = [], [], []

    for i, pair in enumerate(folder_pairs, start=1):
        if isinstance(pair, FolderPair):
            videos_folder, xl_folder, label = pair.videos_folder, pair.xl_folder, pair.label
        else:
            videos_folder, xl_folder, label = pair

        if not os.path.isdir(videos_folder):
            raise FileNotFoundError(f"Videos folder not found: {videos_folder}")
        if not os.path.isdir(xl_folder):
            raise FileNotFoundError(f"Excel folder not found: {xl_folder}")

        videos = [os.path.join(videos_folder, f) for f in os.listdir(videos_folder) if f.endswith(".mp4")]
        xls = [f for f in os.listdir(xl_folder) if f.endswith(".xlsx")]

        print(f"\nPAIR {i} [{label}] Found videos={len(videos)} excels={len(xls)}")

        v, xl_names, xl_paths = match_folder(videos, xls, xl_folder, label=label)
        video_list += v
        xlsx_aligned_names += xl_names
        xl_aligned_paths += xl_paths

    print("\nTOTAL matched videos:", len(video_list))
    print("TOTAL matched excels:", len(xl_aligned_paths))

    powers, zs, z_errs = [], [], []
    for path in xl_aligned_paths:
        powers.append(read_power_mw(path))
        z_mean, z_err = read_z_mean_and_error(path)
        zs.append(z_mean); z_errs.append(z_err)

    df = pd.DataFrame({
        "video_name": [os.path.splitext(os.path.basename(v))[0] for v in video_list],
        "video_path": video_list,
        "xl_name": xlsx_aligned_names,
        "xl_path": xl_aligned_paths,
        "power_mW": powers,
        "z_resolution": zs,
        "z_error": z_errs,
    })

    def make_video_id(path):
        folder = os.path.basename(os.path.dirname(path))
        name = os.path.splitext(os.path.basename(path))[0]
        return f"{folder}\\{name}"

    df["video_id"] = df["video_path"].apply(make_video_id)

    for col in ["z_resolution", "z_error"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df[df["z_resolution"].notna() & (df["z_resolution"] != 0)].copy()
    after = len(df)
    print(f"Dropped {before - after} rows with invalid z resolution (missing / 0 / #DIV/0!). Kept {after} rows.")
    print(f"  (of those kept, {df['z_error'].isna().sum()} missing z_error)")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("Saved dataset to:", out_csv)

    return df


def build_z_dataset_from_xy(
    xy_csv: str = "data/processed/video_xy_dataset.csv",
    out_csv: str = "data/processed/video_z_dataset.csv",
) -> pd.DataFrame:
    """
    Builds the z dataset by reading z_resolution/z_error straight off the
    SAME xl_path already resolved for each of x/y's 111 rows, instead of
    re-running match_folder independently. This guarantees z uses exactly
    x/y's video set (same video_id, same power_mW) rather than its own
    slightly-different valid-row set -- on request, so that x/y's existing
    kfold splits and cached DINOv2 embeddings can be reused for z directly,
    no new video decoding or re-extraction needed.

    A handful of x/y's 111 videos have '#DIV/0!' in their z sheet (a
    formula error in the source workbook, not recoverable) and get dropped
    here -- z ends up with fewer rows than x/y, not more.
    """
    df = pd.read_csv(xy_csv)

    zs, z_errs = [], []
    for path in df["xl_path"]:
        z_mean, z_err = read_z_mean_and_error(path)
        zs.append(z_mean); z_errs.append(z_err)

    out = df[["video_name", "video_path", "xl_name", "xl_path", "power_mW", "video_id"]].copy()
    out["z_resolution"] = pd.to_numeric(pd.Series(zs), errors="coerce")
    out["z_error"] = pd.to_numeric(pd.Series(z_errs), errors="coerce")

    before = len(out)
    out = out[out["z_resolution"].notna() & (out["z_resolution"] != 0)].copy()
    after = len(out)
    print(f"Dropped {before - after} of x/y's {before} rows with invalid z (missing / 0 / #DIV/0!). Kept {after} rows.")
    print(f"  (of those kept, {out['z_error'].isna().sum()} missing z_error)")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out.to_csv(out_csv, index=False)
    print("Saved dataset to:", out_csv)

    return out


def make_z_kfold_splits_from_xy(
    z_csv: str = "data/processed/video_z_dataset.csv",
    xy_split_dir: str = "data/processed/kfold_splits_xy",
    out_dir: str = "data/processed/kfold_splits_z",
    n_splits: int = 5,
) -> List[dict]:
    """
    Reuses x/y's exact fold membership rather than stratifying z from
    scratch: for each of x/y's 5 folds, take its train/val video_id list,
    drop whichever videos don't have a valid z (a handful per fold), and
    attach z_resolution/z_error. Guarantees the same train/val split as
    x/y for every video that's valid in both, and means x/y's cached
    DINOv2 embeddings can be filtered+reused directly for z with zero
    re-extraction (the video/fold membership lines up exactly, just a
    strict subset).
    """
    import json

    z_df = pd.read_csv(z_csv).set_index("video_id")
    os.makedirs(out_dir, exist_ok=True)
    fold_infos = []

    for fold in range(1, n_splits + 1):
        xy_train = pd.read_csv(os.path.join(xy_split_dir, f"fold_{fold}_train.csv"))
        xy_val = pd.read_csv(os.path.join(xy_split_dir, f"fold_{fold}_val.csv"))

        def build_split(xy_split_df):
            valid_ids = [vid for vid in xy_split_df["video_id"] if vid in z_df.index]
            dropped = len(xy_split_df) - len(valid_ids)
            sub = z_df.loc[valid_ids].reset_index()
            return sub, dropped

        train_df, train_dropped = build_split(xy_train)
        val_df, val_dropped = build_split(xy_val)

        train_csv = os.path.join(out_dir, f"fold_{fold}_train.csv")
        val_csv = os.path.join(out_dir, f"fold_{fold}_val.csv")
        split_json = os.path.join(out_dir, f"fold_{fold}_split.json")

        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)

        meta = {
            "fold": fold, "n_splits": n_splits, "source": "reused from kfold_splits_xy",
            "train_video_ids": train_df["video_id"].tolist(),
            "val_video_ids": val_df["video_id"].tolist(),
        }
        with open(split_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"Fold {fold} | train: {len(train_df)} rows ({train_dropped} dropped, no valid z) "
              f"| val: {len(val_df)} rows ({val_dropped} dropped, no valid z)")
        fold_infos.append({"fold": fold, "train_csv": train_csv, "val_csv": val_csv, "split_json": split_json})

    return fold_infos


def make_stratified_kfold_splits(
    in_csv: str,
    out_dir: str,
    n_splits: int = 5,
    n_bins: int = 4,
    seed: int = 42,
    strat_col: str = "log_x_resolution",
    log_transform_col: str = "x_resolution",
):
    """
    Vendored copy of printer_ml's kfold_split.make_stratified_kfold_splits.
    Stratifies on log(x_resolution) by default -- x and y are highly
    correlated, so stratifying on x alone balances the split well for both.
    """
    from sklearn.model_selection import StratifiedKFold
    import json

    df = pd.read_csv(in_csv)

    if "video_id" not in df.columns:
        raise ValueError("Input CSV must contain video_id")
    if strat_col not in df.columns:
        if log_transform_col is None or log_transform_col not in df.columns:
            raise ValueError(f"Input CSV must contain '{strat_col}' or a valid log_transform_col")
        df[strat_col] = np.log(df[log_transform_col].astype(float))

    if len(df) != df["video_id"].nunique():
        raise ValueError("Duplicate video_id values found; this splitter assumes one row per video.")

    bins = pd.qcut(df[strat_col], q=n_bins, duplicates="drop")
    y_strat = bins.cat.codes

    counts = pd.Series(y_strat).value_counts()
    if counts.min() < n_splits:
        raise ValueError(
            f"Cannot do StratifiedKFold with n_splits={n_splits}. "
            f"Smallest bin has only {int(counts.min())} samples."
        )

    os.makedirs(out_dir, exist_ok=True)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_infos = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, y_strat), start=1):
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()

        train_csv = os.path.join(out_dir, f"fold_{fold}_train.csv")
        val_csv = os.path.join(out_dir, f"fold_{fold}_val.csv")
        split_json = os.path.join(out_dir, f"fold_{fold}_split.json")

        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)

        meta = {
            "fold": fold, "seed": seed, "n_splits": n_splits,
            "train_video_ids": train_df["video_id"].tolist(),
            "val_video_ids": val_df["video_id"].tolist(),
        }
        with open(split_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"Fold {fold} | train: {len(train_df)} rows | val: {len(val_df)} rows")
        fold_infos.append({"fold": fold, "train_csv": train_csv, "val_csv": val_csv, "split_json": split_json})

    return fold_infos
