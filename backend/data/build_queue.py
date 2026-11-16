#!/usr/bin/env python3
"""Build the app-id queue (games + DLC) in SQLite. Config from .env."""
import os, sys, time, sqlite3, requests
from dotenv import load_dotenv

load_dotenv()
URL = "https://api.steampowered.com/IStoreService/GetAppList/v1/"
KEY = os.getenv("STEAM_API_KEY")
DB_PATH = os.getenv("DB_PATH", "steam_state.db")
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "50000"))

def init_db(path):
    db = sqlite3.connect(path)
    db.execute("""CREATE TABLE IF NOT EXISTS apps (
        appid INTEGER PRIMARY KEY,
        name TEXT,
        status TEXT NOT NULL DEFAULT 'queue',   -- queue | skipped | fetched
        total_reviews INTEGER,
        updated_at REAL)""")
    db.execute("CREATE INDEX IF NOT EXISTS idx_status ON apps(status, appid)")
    db.commit()
    return db

def main():
    if not KEY:
        sys.exit("STEAM_API_KEY missing from .env")

    db = init_db(DB_PATH)
    last_appid, added, total = 0, 0, 0
    while True:
        params = dict(key=KEY, include_games="true", include_dlc="true",
                      include_software="false", include_videos="false",
                      include_hardware="false", max_results=MAX_RESULTS,
                      last_appid=last_appid)
        try:
            r = requests.get(URL, params=params, timeout=60)
            r.raise_for_status()
            resp = r.json().get("response", {})
        except Exception as e:
            print(f"Error: {e}; retrying in 30s"); time.sleep(30); continue

        apps = resp.get("apps", [])
        for app in apps:
            cur = db.execute(
                "INSERT OR IGNORE INTO apps(appid,name,status,updated_at) VALUES(?,?, 'queue', ?)",
                (app["appid"], app.get("name"), time.time()))
            added += cur.rowcount
        db.commit()
        total += len(apps)
        print(f"page: {len(apps)} apps (total seen {total}, newly queued {added})")

        if not resp.get("have_more_results") or not apps:
            break
        last_appid = resp.get("last_appid", apps[-1]["appid"])

    for s, n in db.execute("SELECT status, COUNT(*) FROM apps GROUP BY status"):
        print(f"{s}: {n}")

if __name__ == "__main__":
    main()