"""Collect legit URLs *with paths* from the Common Crawl index.

Why: PhiUSIIL's legit URLs are bare homepages, so a full-URL model trained on it
learns "has a path = phishing". Common Crawl lists real pages crawled from real
sites, which gives legit URLs that look like real browsing.

What it does:
  1. Picks domains from the Tranco list: half from the top 10k, half from ranks
     10k to 300k (the long tail, where v1 struggles).
  2. Skips benchmark domains, free-hosting platforms, shorteners, and any domain
     that appears in the phishing list (a compromised site isn't "legit").
  3. Asks the Common Crawl index for up to 200 captures per domain, keeps
     successful HTML pages, and samples a few per domain.

It's slow on purpose (about 1 request per second) to be polite to a free
service, and resumable: stop it any time with Ctrl+C and run it again to continue.

Usage (from the project folder, venv active):
  python scripts/collect_commoncrawl.py --domains 20      # quick test first
  python scripts/collect_commoncrawl.py --domains 2500    # the real run, about 1 to 2 hours
"""

from __future__ import annotations

import argparse
import csv
import http.client
import json
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from phishdetect.benchmark import RAW_DIR
from phishdetect.features import SHORTENERS, split_host
from phishdetect.fullurl import COMMONCRAWL_PATH, benchmark_domains, load_phishing_database

UA = "phishdetect-research/0.1 (student ML project; polite crawler)"
DONE_PATH = RAW_DIR / "commoncrawl_done.txt"


def http_get(url: str, timeout: int = 60) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="ignore")


def latest_crawl() -> str:
    return json.loads(http_get("https://index.commoncrawl.org/collinfo.json"))[0]["id"]


class IndexUnreachable(Exception):
    """Raised when the index can't be reached at all for one domain."""


def query_domain(crawl: str, domain: str, limit: int = 100, fetch=None) -> list[str]:
    """Return successful HTML page URLs captured for a domain (and its subdomains)."""
    fetch = fetch or http_get
    # fl= asks only for the 3 fields we use, which makes each response much smaller
    qs = urllib.parse.urlencode({"url": domain, "matchType": "domain", "output": "json",
                                 "limit": limit, "fl": "url,status,mime"})
    url = f"https://index.commoncrawl.org/{crawl}-index?{qs}"
    for attempt in range(4):
        try:
            text = fetch(url)
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:          # the index has no captures for this domain
                return []
            if 400 <= e.code < 500 and e.code != 429:
                # the index refused this one query (e.g. 400 Bad Request on an odd domain
                # name): skip the domain, it says nothing about the others
                print(f"    skipping {domain} (index answered {e.code})", flush=True)
                return []
            if e.code in (429, 500, 502, 503, 504):
                wait = min(60, 5 * 2 ** attempt)
                print(f"    server busy ({e.code}), waiting {wait}s", flush=True)
                time.sleep(wait)
                continue
            wait = min(60, 5 * 2 ** attempt)   # any other server error: wait and retry
            print(f"    server error ({e.code}), waiting {wait}s", flush=True)
            time.sleep(wait)
            continue
        except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
            # dropped connections, timeouts, and responses cut off halfway (IncompleteRead)
            wait = min(60, 5 * 2 ** attempt)
            print(f"    network error ({type(e).__name__}), waiting {wait}s", flush=True)
            time.sleep(wait)
    else:
        raise IndexUnreachable(domain)

    out = []
    for line in text.splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(rec.get("status")) == "200" and "html" in str(rec.get("mime", "")):
            out.append(rec["url"])
    return list(dict.fromkeys(out))   # dedupe, keep order


def pick_domains(n: int, seed: int, exclude: set[str]) -> pd.DataFrame:
    tranco = pd.read_csv(RAW_DIR / "tranco_top1m.csv.zip", names=["rank", "domain"], header=None, nrows=300_000)
    rows = []
    for rank, d in zip(tranco["rank"], tranco["domain"].astype(str)):
        p = split_host(d)
        reg = p["registered"]
        if p["platform"] or reg in SHORTENERS or reg in exclude or not reg:
            continue
        rows.append((rank, reg))
    df = pd.DataFrame(rows, columns=["rank", "domain"]).drop_duplicates("domain")
    head, tail = df[df["rank"] <= 10_000], df[df["rank"] > 10_000]
    k = n // 2
    return pd.concat([head.sample(min(k, len(head)), random_state=seed),
                      tail.sample(min(n - k, len(tail)), random_state=seed)]).sample(frac=1, random_state=seed)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domains", type=int, default=2500, help="how many domains to query")
    ap.add_argument("--per-domain", type=int, default=15, help="max URLs kept per domain")
    ap.add_argument("--crawl", default=None, help="crawl id, e.g. CC-MAIN-2026-40 (default: latest)")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    print("building the exclusion list (benchmark + phishing domains), takes about a minute...", flush=True)
    exclude = benchmark_domains()
    phish = load_phishing_database()
    exclude |= set(phish["url"].map(lambda u: split_host(u)["registered"]))
    print(f"  excluding {len(exclude):,} registered domains", flush=True)

    domains = pick_domains(args.domains, args.seed, exclude)
    done = set(DONE_PATH.read_text().split()) if DONE_PATH.exists() else set()
    todo = domains[~domains["domain"].isin(done)]
    crawl = args.crawl or latest_crawl()
    print(f"crawl {crawl}: {len(domains)} domains picked, {len(done & set(domains['domain']))} already done, "
          f"{len(todo)} to go", flush=True)

    rng = random.Random(args.seed)
    new_file = not COMMONCRAWL_PATH.exists()
    COMMONCRAWL_PATH.parent.mkdir(parents=True, exist_ok=True)
    kept_total, failures_in_a_row = 0, 0
    try:
        with open(COMMONCRAWL_PATH, "a", newline="") as f, open(DONE_PATH, "a") as donef:
            w = csv.writer(f)
            if new_file:
                w.writerow(["url", "domain", "rank", "crawl"])
            for i, (rank, domain) in enumerate(zip(todo["rank"], todo["domain"]), 1):
                try:
                    urls = query_domain(crawl, domain)
                    failures_in_a_row = 0
                except IndexUnreachable:
                    failures_in_a_row += 1
                    print(f"    gave up on {domain} for now (will retry on the next run)", flush=True)
                    if failures_in_a_row >= 5:
                        print("\nCommon Crawl has failed 5 domains in a row: it's probably down or your "
                              "connection dropped. Progress is saved; run the same command later to continue.")
                        sys.exit(1)
                    continue
                keep = rng.sample(urls, min(args.per_domain, len(urls)))
                for u in keep:
                    w.writerow([u, domain, rank, crawl])
                f.flush()
                donef.write(domain + "\n"); donef.flush()
                kept_total += len(keep)
                if i % 25 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)} domains, {kept_total} URLs kept this session", flush=True)
                time.sleep(args.delay)
    except KeyboardInterrupt:
        print("\nstopped. Run the same command again to continue where you left off.")
        sys.exit(0)
    print(f"done. URLs are in {COMMONCRAWL_PATH}")


if __name__ == "__main__":
    main()
