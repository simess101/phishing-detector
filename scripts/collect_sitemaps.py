"""Collect legit URLs with paths from websites' own sitemaps (fast alternative to Common Crawl).

Most sites list their pages in a sitemap (robots.txt says where, usually /sitemap.xml).
Each site gets one to four small requests, and requests go to thousands of different
servers, so it's safe to work on several sites at once. About 2,500 domains take
5 to 10 minutes, compared with hours on the overloaded Common Crawl index.

Domain choice and exclusions are the same as collect_commoncrawl.py (Tranco head + long
tail, no benchmark, platform, shortener, or phishing domains), and domains already
collected from Common Crawl are skipped, so the two sources add up.

Resumable: stop with Ctrl+C and run the same command again.

Usage (from the project folder, venv active):
  python scripts/collect_sitemaps.py --domains 50       # quick test
  python scripts/collect_sitemaps.py --domains 3000     # the real run
"""

from __future__ import annotations

import argparse
import csv
import gzip
import random
import re
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_commoncrawl import DONE_PATH as CC_DONE_PATH, pick_domains  # noqa: E402

from phishdetect.benchmark import RAW_DIR  # noqa: E402
from phishdetect.features import split_host  # noqa: E402
from phishdetect.fullurl import benchmark_domains, load_phishing_database  # noqa: E402

UA = "Mozilla/5.0 (compatible; phishdetect-research/0.1; student ML project)"
OUT_PATH = RAW_DIR / "sitemap_urls.csv"
DONE_PATH = RAW_DIR / "sitemap_done.txt"
MAX_BYTES = 5_000_000          # never read more than 5 MB from one sitemap
_LOC = re.compile(rb"<loc>\s*(.*?)\s*</loc>", re.I | re.S)


def http_get(url: str, timeout: int = 10) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read(MAX_BYTES)
        if r.headers.get("Content-Encoding") == "gzip" or body[:2] == b"\x1f\x8b":
            try:
                body = gzip.decompress(body)
            except (OSError, EOFError):
                return b""
        return body


def sitemaps_from_robots(text: str) -> list[str]:
    return [m.strip() for m in re.findall(r"(?im)^\s*sitemap:\s*(\S+)", text)]


def parse_sitemap(body: bytes) -> tuple[list[str], bool]:
    """Return (urls, is_index). An index lists other sitemaps instead of pages."""
    is_index = b"<sitemapindex" in body[:2000].lower()
    locs = [m.decode("utf-8", "ignore").replace("&amp;", "&") for m in _LOC.findall(body)]
    return locs, is_index


def collect_domain(domain: str, per_domain: int, rng: random.Random, fetch=http_get) -> list[str]:
    """Up to per_domain page URLs on this domain (or its subdomains) from its sitemaps."""
    def safe(url):
        try:
            return fetch(url)
        except Exception:      # any network/HTTP/SSL error: this sitemap just doesn't count
            return b""

    candidates = sitemaps_from_robots(safe(f"https://{domain}/robots.txt").decode("utf-8", "ignore"))
    candidates = candidates[:3] or [f"https://{domain}/sitemap.xml"]

    pages: list[str] = []
    for sm in candidates:
        locs, is_index = parse_sitemap(safe(sm))
        if is_index and locs:                       # follow one child sitemap
            locs, _ = parse_sitemap(safe(rng.choice(locs)))
        pages += locs
        if len(pages) >= per_domain * 5:
            break

    keep = []
    for u in dict.fromkeys(pages):
        if not re.match(r"^https?://", u, re.I) or len(u) > 500:
            continue
        if split_host(u)["registered"] != domain:   # sitemaps sometimes list other sites
            continue
        if re.search(r"\.(xml|gz|jpg|jpeg|png|gif|pdf|webp|svg|mp4|zip)(\?|$)", u, re.I):
            continue
        keep.append(u)
    return rng.sample(keep, min(per_domain, len(keep)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domains", type=int, default=3000)
    ap.add_argument("--per-domain", type=int, default=10)
    ap.add_argument("--workers", type=int, default=16, help="sites fetched at the same time")
    ap.add_argument("--seed", type=int, default=7, help="different from Common Crawl's, for new domains")
    args = ap.parse_args(argv)

    print("building the exclusion list, takes about a minute...", flush=True)
    exclude = benchmark_domains()
    exclude |= set(load_phishing_database()["url"].map(lambda u: split_host(u)["registered"]))
    if CC_DONE_PATH.exists():
        exclude |= set(CC_DONE_PATH.read_text().split())       # already covered by Common Crawl

    domains = pick_domains(args.domains, args.seed, exclude)
    done = set(DONE_PATH.read_text().split()) if DONE_PATH.exists() else set()
    todo = domains[~domains["domain"].isin(done)]
    print(f"{len(domains)} domains picked, {len(todo)} to go, {args.workers} at a time", flush=True)

    new_file = not OUT_PATH.exists()
    lock = threading.Lock()
    kept, with_pages = 0, 0
    with open(OUT_PATH, "a", newline="") as f, open(DONE_PATH, "a") as donef:
        w = csv.writer(f)
        if new_file:
            w.writerow(["url", "domain", "rank", "source"])
        pool = ThreadPoolExecutor(max_workers=args.workers)
        futures = {pool.submit(collect_domain, d, args.per_domain, random.Random(f"{args.seed}{d}")): (r, d)
                   for r, d in zip(todo["rank"], todo["domain"])}
        try:
            for i, fut in enumerate(as_completed(futures), 1):
                rank, domain = futures[fut]
                urls = fut.result()
                with lock:
                    for u in urls:
                        w.writerow([u, domain, rank, "sitemap"])
                    f.flush()
                    donef.write(domain + "\n"); donef.flush()
                kept += len(urls); with_pages += bool(urls)
                if i % 100 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)} domains, {with_pages} had sitemaps, {kept} URLs kept", flush=True)
        except KeyboardInterrupt:
            print("\nstopping (finishing requests already in flight)...")
            pool.shutdown(wait=False, cancel_futures=True)
            print("stopped. Run the same command again to continue.")
            sys.exit(0)
        pool.shutdown()
    print(f"done. URLs are in {OUT_PATH}")


if __name__ == "__main__":
    main()
