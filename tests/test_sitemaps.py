import gzip
import importlib.util
import random
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _mod():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("collect_sitemaps", SCRIPTS / "collect_sitemaps.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_robots_parsing():
    m = _mod()
    text = "User-agent: *\nDisallow: /x\nSitemap: https://a.com/sm1.xml\nsitemap:https://a.com/sm2.xml.gz\n"
    assert m.sitemaps_from_robots(text) == ["https://a.com/sm1.xml", "https://a.com/sm2.xml.gz"]


def test_parse_sitemap_and_index():
    m = _mod()
    urlset = b'<urlset><url><loc>https://a.com/p?x=1&amp;y=2</loc></url><url><loc> https://a.com/q </loc></url></urlset>'
    assert m.parse_sitemap(urlset) == (["https://a.com/p?x=1&y=2", "https://a.com/q"], False)
    index = b'<?xml version="1.0"?><sitemapindex><sitemap><loc>https://a.com/s1.xml</loc></sitemap></sitemapindex>'
    assert m.parse_sitemap(index) == (["https://a.com/s1.xml"], True)


def test_collect_domain_follows_index_and_filters():
    m = _mod()
    pages = b"<urlset>" + b"".join(
        f"<url><loc>https://a.com/page{i}</loc></url>".encode() for i in range(30)) + (
        b"<url><loc>https://other.com/x</loc></url><url><loc>https://a.com/img.png</loc></url>"
        b"<url><loc>https://blog.a.com/post</loc></url></urlset>")
    site = {
        "https://a.com/robots.txt": b"Sitemap: https://a.com/index.xml",
        "https://a.com/index.xml": b"<sitemapindex><sitemap><loc>https://a.com/pages.xml.gz</loc></sitemap></sitemapindex>",
        "https://a.com/pages.xml.gz": pages,
    }
    def fake(url):
        if url not in site:
            raise OSError("404")
        return site[url]
    out = m.collect_domain("a.com", 10, random.Random(0), fetch=fake)
    assert len(out) == 10
    assert all(u.startswith(("https://a.com/page", "https://blog.a.com/")) for u in out)


def test_collect_domain_no_robots_falls_back_and_survives_errors():
    m = _mod()
    def fake(url):
        if url == "https://b.org/sitemap.xml":
            return b"<urlset><url><loc>https://b.org/about</loc></url></urlset>"
        raise TimeoutError("slow site")
    assert m.collect_domain("b.org", 5, random.Random(0), fetch=fake) == ["https://b.org/about"]
    assert m.collect_domain("dead.net", 5, random.Random(0), fetch=lambda u: (_ for _ in ()).throw(OSError())) == []


def test_load_legit_combines_sources(tmp_path, monkeypatch):
    from phishdetect import fullurl
    (tmp_path / "cc.csv").write_text("url,domain,rank,crawl\nhttps://a.com/x,a.com,1,CC\n")
    (tmp_path / "sm.csv").write_text("url,domain,rank,source\nhttps://b.org/y,b.org,2,sitemap\nhttps://a.com/x,a.com,1,sitemap\n")
    monkeypatch.setattr(fullurl, "COMMONCRAWL_PATH", tmp_path / "cc.csv")
    monkeypatch.setattr(fullurl, "SITEMAP_PATH", tmp_path / "sm.csv")
    df = fullurl.load_legit()
    assert sorted(df["url"]) == ["https://a.com/x", "https://b.org/y"]
    assert set(df["source"]) == {"commoncrawl", "sitemap"}
