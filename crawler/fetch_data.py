#!/usr/bin/env python3
"""Per app: 1x IUserReviewsService/GetAppReviews (count + top English reviews, all dates),
then 1x appdetails if total_reviews >= MIN_REVIEWS. Config from .env."""
import json, os, time, sqlite3, requests
from dotenv import load_dotenv

load_dotenv()
REVIEWS_URL = "https://api.steampowered.com/IUserReviewsService/GetAppReviews/v1/"
DETAILS_URL = "https://store.steampowered.com/api/appdetails"

KEY = os.getenv("STEAM_API_KEY")
DB_PATH = os.getenv("DB_PATH", "steam_state.db")
OUT = os.getenv("OUTPUT_DIR", "steam_data")
INTERVAL = float(os.getenv("REQUEST_INTERVAL", "2"))
print(INTERVAL)
BACKOFF = float(os.getenv("RATE_LIMIT_WAIT", "180"))
MIN_REVIEWS = int(os.getenv("MIN_REVIEWS", "100"))
TOP_N = int(os.getenv("TOP_N", "10"))
LIMIT = int(os.getenv("LIMIT", "0"))

class Throttle:
    def __init__(self, interval, backoff):
        self.interval, self.backoff, self.last = interval, backoff, 0.0

    def get(self, url, params, rate_codes):
        while True:
            wait = self.interval - (time.time() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()
            try:
                r = requests.get(url, params=params, timeout=30)
            except requests.RequestException as e:
                print(f"  network error ({e}); waiting {self.backoff}s")
                time.sleep(self.backoff); continue
            if r.status_code in rate_codes:
                print(f"  rate limited (HTTP {r.status_code}); waiting {self.backoff}s")
                time.sleep(self.backoff); continue
            r.raise_for_status()
            try:
                return r.json()
            except ValueError:
                print(f"  non-JSON response; waiting {self.backoff}s")
                time.sleep(self.backoff)

def get_reviews(t, appid):
    payload = {
        "appid": appid,
        "filter": 0,                  # Helpful
        "day_range": 0,               # 0 = no limit (all dates)
        "languages": ["english"],
        "review_type": 0,
        "purchase_type": 1,
        "num_per_page": TOP_N,
        "cursor": "*",
        "filter_offtopic_activity": True,
    }
    params = {"input_json": json.dumps(payload)}
    if KEY:
        params["key"] = KEY
    return t.get(REVIEWS_URL, params, {429})

def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def process(appid, t):
    resp = get_reviews(t, appid)
    body = resp.get("response", resp)
    total = int(body.get("query_summary", {}).get("total_reviews", 0) or 0)
    if total < MIN_REVIEWS:
        return "skipped", total

    if body.get("day_range_used", 0) != 0:
        print(f"  warning: day_range_used={body.get('day_range_used')} (expected 0)")

    details = t.get(DETAILS_URL, dict(appids=appid, l="english"), {429, 403})
    save_json(os.path.join(OUT, "reviews", f"{appid}.json"), resp)
    save_json(os.path.join(OUT, "details", f"{appid}.json"), details)
    return "fetched", total

def main():
    os.makedirs(os.path.join(OUT, "reviews"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "details"), exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    t = Throttle(INTERVAL, BACKOFF)
    failures, done = {}, 0

    while True:
        skip = tuple(failures) or (-1,)
        row = db.execute(
            f"SELECT appid, name FROM apps WHERE status='queue' "
            f"AND appid NOT IN ({','.join('?'*len(skip))}) ORDER BY appid LIMIT 1", skip).fetchone()
        if not row:
            break
        appid, name = row
        try:
            status, total = process(appid, t)
        except Exception as e:
            failures[appid] = str(e)
            print(f"[{appid}] {name}: error {e} (left in queue)")
            continue
        db.execute("UPDATE apps SET status=?, total_reviews=?, updated_at=? WHERE appid=?",
                   (status, total, time.time(), appid))
        db.commit()
        done += 1
        print(f"[{appid}] {name}: {status} ({total} reviews)")
        if LIMIT and done >= LIMIT:
            break

    for s, n in db.execute("SELECT status, COUNT(*) FROM apps GROUP BY status"):
        print(f"{s}: {n}")
    if failures:
        print(f"{len(failures)} apps errored this run and remain in queue")

if __name__ == "__main__":
    main()