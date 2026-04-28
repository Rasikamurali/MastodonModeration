# waybackmachine_preelon.py
#
# Collects post-Elon (Jan-Feb 2024) Mastodon instance rules from Wayback
# Machine snapshots. Focuses specifically on the Jan-Feb 2024 window for
# instances already captured pre-Elon, enabling longitudinal comparison.
# Uses CDX API with retry logic and falls back to the availability API.
#
# Input:  non_personal_preelon_notnull.csv  (instances with valid pre-Elon snapshots)
# Output: non_personal_postelon_2024.csv    (Jan-Feb 2024 rules per instance)

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry
import time
import csv
import json
import os
import re

# Wayback CDX and snapshot fetch endpoints
CDX_URL = "https://web.archive.org/cdx/search/cdx"
WAYBACK_PREFIX = "https://web.archive.org/web"
WAYBACK_AVAILABLE = "https://archive.org/wayback/available"
UA = "MastoSnapshotCheck/0.2 (+your-email@example.com)"
HEADERS = {"User-Agent": UA}


# --- session with retries + backoff ---
def make_session():
    s = requests.Session()
    s.headers.update({"User-Agent": UA})
    retries = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    return s


session = make_session()


def fallback_available_api(url, date="20240130"):
    params = {"url": url, "timestamp": date}
    try:
        res = session.get(WAYBACK_AVAILABLE, params=params, headers=HEADERS, timeout=30)
        res.raise_for_status()
        data = res.json()
        snap = data.get("archived_snapshots", {}).get("closest")
        if snap and snap.get("available") and snap.get("status") == "200":
            return snap["timestamp"]
    except Exception as e:
        print(f"Fallback failed: {e}")
    return None


def try_fetch_snapshot(ts, url):
    wayback_url = f"{WAYBACK_PREFIX}/{ts}id_/{url}"
    try:
        res = session.get(wayback_url, headers=HEADERS, timeout=30)
        if res.status_code != 200 or not res.text.strip():
            return None

        # JSON attempt
        try:
            return res.json()
        except Exception:
            pass

        # look for HTML meta redirect
        m = re.search(r'URL=([^\"]+)', res.text, re.IGNORECASE)
        if m:
            redirect_target = m.group(1)
            if redirect_target.startswith("/web/"):
                redirect_url = "https://web.archive.org" + redirect_target
                r2 = session.get(redirect_url, headers=HEADERS, timeout=30)
                if r2.status_code == 200 and r2.text.strip():
                    try:
                        return r2.json()
                    except:
                        return None
        return None
    except Exception as e:
        print(f"Error fetching {wayback_url}: {e}")
        return None


def find_snapshot_in_window(url, start_ts="20240101000000", end_ts="20240228235959", retries=5):
    params = {
        "url": url,
        "matchType": "exact",
        "output": "json",
        "fl": "timestamp,statuscode",
        "from": start_ts[:8],   # YYYYMMDD
        "to": end_ts[:8],       # YYYYMMDD
        "limit": "200",
        "sort": "ascending",
    }

    for attempt in range(retries):
        try:
            res = session.get(CDX_URL, params=params, headers=HEADERS, timeout=60)
            res.raise_for_status()
            if not res.text.strip():
                print("CDX empty body → fallback to blue-dot")
                return fallback_available_api(url, start_ts), "available-blue-dot"

            data = res.json()
            if len(data) > 1:
                for row in data[1:]:
                    ts = row[0]
                    snap_data = try_fetch_snapshot(ts, url)
                    if snap_data and isinstance(snap_data, dict) and "rules" in snap_data:
                        return ts, "cdx-window"

            return fallback_available_api(url, start_ts), "available-blue-dot"

        except Exception as e:
            print(f"Error finding snapshot: {e}")
            return fallback_available_api(url, start_ts), "available-blue-dot"

def collect_instance_jan_feb_2024(host):
    api_path = "/api/v1/instance"
    full_url = f"https://{host}{api_path}"

    ts, source = find_snapshot_in_window(full_url)

    if not ts:
        return {
            "jan-feb2024": {
                "timestamp": None,
                "instance_name": None,
                "rules": None,
                "note": "no valid snapshot Jan-Feb 2024",
            }
        }

    data = try_fetch_snapshot(ts, full_url)

    if not data or not isinstance(data, dict):
        return {
            "jan-feb2024": {
                "timestamp": ts,
                "instance_name": None,
                "rules": None,
                "note": f"snapshot found ({source}) but no valid JSON",
            }
        }

    return {
        "jan-feb2024": {
            "timestamp": ts,
            "instance_name": data.get("title"),
            "rules": data.get("rules", []),
            "note": source,
        }
    }


def write_instance_to_csv(instance: str, results: dict, outfile: str):
    is_new_file = not os.path.exists(outfile)

    with open(outfile, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(["instance", "period", "timestamp", "instance_name", "rules", "note"])

        for period, values in results.items():
            writer.writerow(
                [
                    instance,
                    period,
                    values.get("timestamp"),
                    values.get("instance_name"),
                    json.dumps(values.get("rules"), ensure_ascii=False),
                    values.get("note"),
                ]
            )


# --- Example usage ---
if __name__ == "__main__":

    # popular_instances = pd.read_csv(r'data/community_rules_data.csv')
    # print(popular_instances.columns)
    # popular_instances = popular_instances.drop_duplicates(subset=['Instance Name'])

    # bins = [1, 10, 100, 1000, 10000, 100000, 1000000]  
    # bin_labels = [r"$10^{1}$", r"$10^{2}$", r"$10^{3}$", r"$10^{4}$", r"$10^{5}$", r"$10^{6}$"]


    # # Assign each instance to a user count bin
    # popular_instances['User Count Bin'] = pd.cut(popular_instances['User Count'], bins=bins, labels=bin_labels, right=False)

    # # Find the minimum group size (so sampling is fair across all groups)
    # min_size = popular_instances['User Count Bin'].value_counts().min()
    # print("Minimum group size:", min_size)

    # # Sample equal number of instances from each group
    # sampled_instances = (
    #     popular_instances.groupby('User Count Bin', group_keys=False)
    #     .apply(lambda x: x.sample(n=min_size, random_state=42))  # random_state for reproducibility
    # )

    # print("Sampled shape:", sampled_instances.shape)
    # print(sampled_instances['User Count Bin'].value_counts())

    # # Get the names as a list if needed
    # popular_instance_names = sampled_instances['Instance Name'].tolist()

    # print(len(popular_instance_names))
    # print(popular_instance_names[:10])

    # df = pd.read_csv(r'data\sampled_annotations_GPT_category(4)_full.csv')
    # print(df.columns)
    # df = df.drop_duplicates(subset=['Instance Name'])
    # print(len(df))
    # # Only get instance with topic != personal
    # GPT_category1 = []
    # for row in df['GPT category']: 
    #     GPT_category1.append(re.sub(r'A:\s*|[\[\]\'"]', '', row).strip())
        
    # df['GPT category cleaned'] = GPT_category1
    # print(df['GPT category cleaned'].unique())
    # df = df[~df['GPT category cleaned'].str.contains(r'\bPersonal\b', case=False, na=False)]


    # print(len(df))
    # popular_instance_names = df['Instance Name'].tolist()
    # popular_instance_names = popular_instance_names[1426:]
    # print(len(popular_instance_names))

    df = pd.read_csv('non_personal_preelon_notnull.csv')
    print(len(df))
    print(df.columns)
    popular_instance_names = df['instance'].tolist()
    print(len(popular_instance_names))

    outfile = "non_personal_postelon_2024.csv"
    for instance in popular_instance_names:
        print(f"Fetching {instance}")
        result = collect_instance_jan_feb_2024(instance)
        write_instance_to_csv(instance, result, outfile)
        time.sleep(2)  # polite delay