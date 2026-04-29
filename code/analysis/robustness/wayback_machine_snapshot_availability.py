# wayback_machine_scraping.py
#
# Scrapes Mastodon instance rules from Wayback Machine snapshots across four
# time periods: Sep 2022 (pre-Elon), Jan 2023, Jan 2024, and Oct 2024.
# Uses the CDX API to find usable snapshots, then fetches /api/v1/instance
# JSON from those snapshots. Results are appended row-by-row to a CSV.
#
# Input:  data/community_rules_data.csv  (main instance list for stratified sampling)
# Output: sampled_instances_norm_history_preelon.csv
#         (per-instance rules across four Wayback time periods)

import urllib.parse
import urllib.request
import json
import pandas as pd
import numpy as np
from collections import Counter
import requests
from requests.adapters import HTTPAdapter, Retry
import time
import csv
import json
import os

# Wayback Machine CDX API endpoint
CDX_URL = "https://web.archive.org/cdx/search/cdx"
UA = "MastoSnapshotCheck/0.1 (+your-email@example.com)"

CDX_URL = "https://web.archive.org/cdx/search/cdx"
UA = "MastoSnapshotCheck/0.2 (+your-email@example.com)"

# --- session with retries + backoff ---
def make_session():
    s = requests.Session()
    s.headers.update({"User-Agent": UA})
    retries = Retry(
        total=5,                # total attempts per request
        connect=5,
        read=5,
        backoff_factor=0.8,     # 0.8, 1.6, 3.2, ...
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    return s

session = make_session()

def has_snapshots(host: str, timeout=20) -> bool:
    """
    Returns True if Wayback has any capture for host (any status code).
    Retries on transient errors and surfaces readable exceptions.
    """
    params = {
        "url": host,
        "matchType": "host",
        "output": "json",
        "limit": "1",
    }
    try:
        r = session.get(CDX_URL, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return bool(data)
    except requests.exceptions.ProxyError as e:
        print(f"[{host}] Proxy error: {e}. If you're behind a proxy, set HTTPS_PROXY/HTTP_PROXY env vars.", flush=True)
        return False
    except requests.exceptions.SSLError as e:
        print(f"[{host}] SSL error: {e}. If you have SSL interception, try system certs or disable interception.", flush=True)
        return False
    except requests.exceptions.ConnectTimeout:
        print(f"[{host}] Connect timeout. Network or firewall may be blocking archive.org.", flush=True)
        return False
    except requests.exceptions.ReadTimeout:
        print(f"[{host}] Read timeout. Try increasing timeout.", flush=True)
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"[{host}] Connection error: {e}", flush=True)
        return False
    except json.JSONDecodeError as e:
        print(f"[{host}] Bad JSON from CDX: {e}", flush=True)
        return False

# Example usage

# instance_list = pd.read_csv(r'data/community_rules_data.csv')
# print(instance_list.columns)
# domains = instance_list['Instance Name'].tolist()
# domains = list(set(domains))
# domains = domains[5000:]

# records = []
# for d in domains:
#     available = "yes" if has_snapshots(d) else "no"
#     records.append({"Instance Name": d, "Available": available})
#     time.sleep(0.2)

# # Make a DataFrame
# df = pd.DataFrame(records)

# # Show the DataFrame
# print(df)

# # Optionally: save to CSV
# df.to_csv("snapshot_availability_5000_6000.csv", index=False)



# def first_and_last_snapshots(host: str):
#     """Return (first_ts, last_ts) or (None, None) if no snapshots exist."""
#     base = {
#         "url": host,
#         "matchType": "host",
#         "output": "json",
#         "fl": "timestamp",
#         "limit": "1",
#     }

#     # First snapshot (ascending)
#     asc = base.copy()
#     asc["sort"] = "ascending"
#     print("ascending")
#     url1 = f"{CDX_URL}?{urllib.parse.urlencode(asc)}"
#     req1 = urllib.request.Request(url1, headers={"User-Agent": UA})
#     try:
#         with urllib.request.urlopen(req1, timeout=30) as resp:
#             data = json.loads(resp.read().decode("utf-8"))
#         if not data:
#             return None, None
#         first_ts = data[1][0]
#         print(first_ts)
#     except Exception:
#         return None, None

#     # Last snapshot (descending)
#     desc = base.copy()
#     print("descending")
#     desc["sort"] = "descending"
#     url2 = f"{CDX_URL}?{urllib.parse.urlencode(desc)}"
#     req2 = urllib.request.Request(url2, headers={"User-Agent": UA})
#     try:
#         with urllib.request.urlopen(req2, timeout=30) as resp:
#             data = json.loads(resp.read().decode("utf-8"))
#             print(data)
#         last_ts = data[0][1] if data else None
#         print(data)
#         print(data[0])
#         print(data[1])
#     except Exception:
#         last_ts = None

#     return first_ts, last_ts


# # Example usage
# domains = ["mastodon.social", "fosstodon.org", "example.com"]

# for d in domains:
#     first, last = first_and_last_snapshots(d)
#     print(first, last)
#     if first and last:
#         print(f"{d} -> YES (first: {first}, last: {last})")
#     else:
#         print(f"{d} -> NO snapshots")



import requests

import requests
import time
import csv
import json
import os

# Wayback CDX and snapshot API
CDX_URL = "https://web.archive.org/cdx/search/cdx"
WAYBACK_PREFIX = "https://web.archive.org/web"
HEADERS = {"User-Agent": "MastoRulesFetcher/0.1"}

# Periods to check (snapshot window)
PERIODS = {
    "2022-09": ("20220901000000", "20220930235959"),
    "2023-01": ("20230101000000", "20230131235959"),
    "2024-01": ("20240101000000", "20240131235959"),
    "2024-10": ("20241001000000", "20241031235959"),
}

import random
WAYBACK_AVAILABLE = "https://archive.org/wayback/available"
 
def fallback_available_api(url, date):
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


# def find_snapshot_timestamp(url, start_ts, end_ts):
#     params = {
#         "url": url,
#         "matchType": "exact",
#         "output": "json",
#         "fl": "timestamp,statuscode",
#         "from": start_ts[:8],
#         "to": end_ts[:8],
#         "limit": "100",
#         "sort": "ascending"
#     }

#     try:
#         res = session.get(CDX_URL, params=params, headers=HEADERS, timeout=60)
#         print(res)
#         res.raise_for_status()
#         if not res.text.strip():
#             print("CDX empty body → fallback")
#             return fallback_available_api(url, start_ts)
#         data = res.json()
#         print(data)

#         if len(data) > 1:
#             print(f"Got {len(data)-1} snapshots for {url} in {start_ts[:6]}")
#             for row in data[1:]:
#                 if len(row) >= 2 and row[1] == "200":
#                     return row[0]
#         print("No 200 in CDX → fallback")
#         return fallback_available_api(url, start_ts)


#     except Exception as e:
#         print(f"Error finding snapshot: {e}")
#         return fallback_available_api(url, start_ts)

import re

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


def find_snapshot_timestamp(url, start_ts, end_ts, limit=200):
    params = {
        "url": url,
        "matchType": "exact",
        "output": "json",
        "fl": "timestamp,statuscode",
        "from": start_ts[:8],
        "to": end_ts[:8],
        "limit": str(limit),
        "sort": "ascending"
    }

    try:
        res = session.get(CDX_URL, params=params, headers=HEADERS, timeout=60)
        res.raise_for_status()
        if not res.text.strip():
            print("CDX empty body → fallback to closest")
            return fallback_available_api(url, start_ts)

        data = res.json()
        if len(data) > 1:
            print(f"Got {len(data)-1} snapshots for {url} in {start_ts[:6]}")
            for row in data[1:]:
                ts = row[0]
                data_json = try_fetch_snapshot(ts, url)
                if data_json and isinstance(data_json, dict) and "rules" in data_json:
                    return ts  # ✅ usable JSON found

        print("No usable snapshot in month → fallback to closest")
        return fallback_available_api(url, start_ts)

    except Exception as e:
        print(f"Error finding snapshot: {e} → fallback to closest")
        return fallback_available_api(url, start_ts)


def fetch_archived_instance_data(timestamp, url):
    wayback_url = f"{WAYBACK_PREFIX}/{timestamp}id_/{url}"
    try:
        res = requests.get(wayback_url, headers=HEADERS, timeout=20)
        if res.status_code == 200 and res.text.strip():
            return res.json()
    except Exception as e:
        print(f"Error fetching {wayback_url}: {e}")
    return None

def get_instance_norms(host):
    results = {}
    api_path = "/api/v1/instance"
    full_url = f"https://{host}{api_path}"

    for label, (start_ts, end_ts) in PERIODS.items():
        ts = find_snapshot_timestamp(full_url, start_ts, end_ts)
        if not ts or not isinstance(ts, str) or not ts.isdigit():
            results[label] = {
                "timestamp": None,
                "instance_name": None,
                "rules": None,
                "note": "no valid snapshot timestamp"
            }
            continue

        print(f"Fetching snapshot: {WAYBACK_PREFIX}/{ts}id_/{full_url}")
        data = fetch_archived_instance_data(ts, full_url)

        if not data or not isinstance(data, dict):
            results[label] = {
                "timestamp": ts,
                "instance_name": None,
                "rules": None,
                "note": "no data or bad format"
            }
            continue

        results[label] = {
            "timestamp": ts,
            "instance_name": data.get("title"),
            "rules": data.get("rules", []),
            "note": None
        }

    return results

def write_instance_to_csv(instance: str, results: dict, outfile: str):
    is_new_file = not os.path.exists(outfile)

    with open(outfile, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(["instance", "period", "timestamp", "instance_name", "rules"])

        for period, values in results.items():
            writer.writerow([
                instance,
                period,
                values.get("timestamp"),
                values.get("instance_name"),
                json.dumps(values.get("rules"), ensure_ascii=False)
            ])

# --- Example usage ---
if __name__ == "__main__":
    domain = "mastodon.ie"  # Replace with any Mastodon instancehttps://mstdn.social/

    # instances = pd.read_csv(r'data/instance_topics_translated.csv')
    # print(instances.columns)
    # instance_name = instances['Instance Name'].tolist()
    # print(len(instance_name))

    # domains = instance_name[:10]
    # print(domains)

    popular_instances = pd.read_csv(r'data/primary/community_rules_data.csv')
    print(popular_instances.columns)
    popular_instances = popular_instances.drop_duplicates(subset=['Instance Name'])

    bins = [1, 10, 100, 1000, 10000, 100000, 1000000]  
    bin_labels = [r"$10^{1}$", r"$10^{2}$", r"$10^{3}$", r"$10^{4}$", r"$10^{5}$", r"$10^{6}$"]


    # Assign each instance to a user count bin
    popular_instances['User Count Bin'] = pd.cut(popular_instances['User Count'], bins=bins, labels=bin_labels, right=False)

    # Find the minimum group size (so sampling is fair across all groups)
    min_size = popular_instances['User Count Bin'].value_counts().min()
    print("Minimum group size:", min_size)

    # Sample equal number of instances from each group
    sampled_instances = (
        popular_instances.groupby('User Count Bin', group_keys=False)
        .apply(lambda x: x.sample(n=min_size, random_state=42))  # random_state for reproducibility
    )

    print("Sampled shape:", sampled_instances.shape)
    print(sampled_instances['User Count Bin'].value_counts())

    # Get the names as a list if needed
    popular_instance_names = sampled_instances['Instance Name'].tolist()

    print(len(popular_instance_names))
    print(popular_instance_names[:10])
    # result = get_instance_norms(domain)
    # write_instance_to_csv(domain, result, "sampled_instances_norm_history_preelon.csv")


    # instances = [
    #     "mastodon.social",
    #     "fosstodon.org",
    #     "techhub.social",
    #     # ... potentially thousands more
    # ]

    for instance in popular_instance_names:
        print(f"Fetching: {instance}")
        try:
            result = get_instance_norms(instance)
            write_instance_to_csv(instance, result, "sampled_instances_norm_history_preelon.csv")
            time.sleep(2)
        except Exception as e:
            print(f"Failed for {instance}: {e}")
