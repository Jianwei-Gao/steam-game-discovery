#!/usr/bin/env python3
"""
steam_to_parquet.py

Turn a folder of Steam description JSONs and a folder of Steam review JSONs
into two Parquet files.

Dependencies:
    pip install beautifulsoup4 lxml pandas pyarrow

Usage:
    python steam_to_parquet.py \
        --desc-dir description --review-dir review \
        --desc-out descriptions.parquet --review-out reviews.parquet \
        --workers 4
"""

import argparse
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

# ---- BeautifulSoup with lxml preferred, html.parser fallback ----------------
try:
    from bs4 import BeautifulSoup
    try:
        BeautifulSoup("<p>x</p>", "lxml")
        BS_PARSER = "lxml"
    except Exception:
        BS_PARSER = "html.parser"
except ImportError:
    print("Missing dependency: pip install beautifulsoup4 lxml", file=sys.stderr)
    sys.exit(1)

import pandas as pd

# ---- HTML filtering configuration -----------------------------------------
# Tags whose entire subtree is embedded media / non-description noise.
STRIP_TAGS = [
    "script", "style", "noscript", "link", "meta", "head",
    "video", "audio", "source", "track", "img", "picture",
    "svg", "canvas", "iframe", "embed", "object",
    "button", "input", "form", "select", "option",
]

URL_PATTERN = re.compile(r"https?://\S+")


def clean_html(html_str: str, keep_unicode_letters: bool = True) -> str:
    """
    Return a single-line, alphanumeric-only version of a chunk of
    Steam-flavoured HTML.

    - Embedded media / non-description tags are dropped entirely.
    - The text inside <p>, <h1..h6>, <ul>/<li>, <b>, <i>, … is kept.
    - Literal URLs are removed.
    - Every non-alphanumeric, non-whitespace character is replaced by a
      space (so word boundaries survive: "Hello,world" -> "Hello world").
    - All whitespace runs (\n, \r\n, \t, multiple spaces) collapse to
      a single space, and the result is stripped.
    """
    if not html_str or not isinstance(html_str, str):
        return ""

    try:
        soup = BeautifulSoup(html_str, BS_PARSER)
    except Exception:
        return html_str

    for tag in soup.find_all(STRIP_TAGS):
        tag.decompose()

    text = soup.get_text(separator=" ")

    # Strip URLs *before* the alnum filter — otherwise "https" survives
    # as a word and the rest of the URL leaves stray fragments.
    text = URL_PATTERN.sub(" ", text)
    text = re.sub(r"['’`´]", "", text)          # apostrophes removed
    
    # Replace every non-alnum, non-whitespace char with a space so that
    # punctuation becomes a word separator rather than a word-joiner.
    if keep_unicode_letters:
        text = "".join(c if (c.isalnum() or c.isspace()) else " " for c in text)
    else:
        text = "".join(
            c if (c.isascii() and c.isalnum()) or c.isspace() else " "
            for c in text
        )

    # Collapse all whitespace (newlines included) to single spaces.
    text = re.sub(r"\s+", " ", text).lower().strip()
    return text

# ---- Per-file processors ---------------------------------------------------
def process_description_file(filepath: str):
    """Return a list of row dicts (one per appid entry in the file)."""
    rows = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[desc] could not read {filepath}: {e}", file=sys.stderr)
        return rows

    if not isinstance(data, dict):
        return rows

    for appid_str, entry in data.items():
        if not (isinstance(appid_str, str) and appid_str.isdigit()):
            continue
        if not isinstance(entry, dict) or not entry.get("success"):
            continue

        d = entry.get("data") or {}
        name  = d.get("name") or ""
        about = clean_html(d.get("about_the_game") or "")
        short = clean_html(d.get("short_description") or "")

        if short and about:
            if short == about: 
                merged = about 
            else:
                merged = f"{short} {about}"
        else:
            merged = short or about

        rows.append({
            "appid": int(appid_str),
            "name": name,
            "about_the_game": about,
            "short_description": short,
            "merged_description": merged,
        })
    return rows


def _extract_appid(filepath: str) -> int | None:
    """Review files are named {appid}.json, so the stem IS the appid."""
    stem = Path(filepath).stem
    if not stem.isdigit():
        print(f"[rev] filename is not a numeric appid: {filepath}", file=sys.stderr)
        return None
    return int(stem)


def process_review_file(filepath: str):
    appid = _extract_appid(filepath)
    if appid is None:
        return None

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[rev] could not read {filepath}: {e}", file=sys.stderr)
        return None

    response = data.get("response") or {}
    summary  = response.get("query_summary") or {}
    reviews  = response.get("reviews") or []

    row = {
        "appid": appid,
        "review_score": summary.get("review_score"),
        "review_score_desc": summary.get("review_score_desc"),
    }
    for i, rev in enumerate(reviews, start=1):
        row[f"review_{i}"] = clean_html(rev.get("review") or "")
    return row


# ---- Driver ---------------------------------------------------------------
def _iter_json_files(directory: str):
    d = Path(directory)
    if not d.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    yield from d.glob("*.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--desc-dir",  default="steam_data/details")
    ap.add_argument("--review-dir", default="steam_data/reviews")
    ap.add_argument("--desc-out",   default="steam_data/descriptions.parquet")
    ap.add_argument("--review-out", default="steam_data/reviews.parquet")
    ap.add_argument("--workers", type=int, default=1,
                    help="multiprocessing workers (1 = single process)")
    args = ap.parse_args()

    # ---------------- Descriptions -------------------------------------
    desc_files = list(_iter_json_files(args.desc_dir))
    print(f"Description files: {len(desc_files)}")

    desc_rows = []
    if args.workers > 1:
        with Pool(args.workers) as pool:
            for i, rows in enumerate(
                pool.imap_unordered(process_description_file,
                                    desc_files, chunksize=32)):
                desc_rows.extend(rows)
                if (i + 1) % 2000 == 0:
                    print(f"  desc: {i+1}/{len(desc_files)} files, "
                          f"{len(desc_rows)} rows")
    else:
        for i, fp in enumerate(desc_files):
            desc_rows.extend(process_description_file(fp))
            if (i + 1) % 2000 == 0:
                print(f"  desc: {i+1}/{len(desc_files)} files, "
                      f"{len(desc_rows)} rows")

    df_desc = pd.DataFrame(desc_rows)
    if "appid" in df_desc.columns:
        df_desc = df_desc.drop_duplicates(subset="appid", keep="first")
    df_desc.to_parquet(args.desc_out, index=False)
    print(f"✔ Wrote {len(df_desc):,} description rows → {args.desc_out}")

    del desc_rows, df_desc

    # ---------------- Reviews ------------------------------------------
    review_files = list(_iter_json_files(args.review_dir))
    print(f"Review files: {len(review_files)}")

    review_rows = []
    if args.workers > 1:
        with Pool(args.workers) as pool:
            for i, row in enumerate(
                pool.imap_unordered(process_review_file,
                                    review_files, chunksize=32)):
                if row is not None:
                    review_rows.append(row)
                if (i + 1) % 2000 == 0:
                    print(f"  rev: {i+1}/{len(review_files)} files, "
                          f"{len(review_rows)} rows")
    else:
        for i, fp in enumerate(review_files):
            row = process_review_file(fp)
            if row is not None:
                review_rows.append(row)
            if (i + 1) % 2000 == 0:
                print(f"  rev: {i+1}/{len(review_files)} files, "
                      f"{len(review_rows)} rows")

    df_rev = pd.DataFrame(review_rows)
    if "appid" in df_rev.columns:
        df_rev = df_rev.drop_duplicates(subset="appid", keep="first")
    df_rev.to_parquet(args.review_out, index=False)
    print(f"✔ Wrote {len(df_rev):,} review rows → {args.review_out}")


if __name__ == "__main__":
    main()