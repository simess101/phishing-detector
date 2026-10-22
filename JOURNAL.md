# Project Journal

A running log of what I tried, what I found, and why I made each change. Newest entries go at the bottom.

**Template for new entries**

```
## YYYY-MM-DD · Short title

**Goal:** what I set out to do.
**What happened:** what I ran and what I saw (numbers, screenshots, errors).
**Why it matters / what I think is going on:**
**Decision:** what I changed or will do next, and why.
```

---

## 2026-10-22 · Project setup

**Goal:** Get a reproducible Python project running on my laptop and pushed to GitHub.

**What happened:** The repo has a `src/phishdetect` package (feature extraction and data loading), unit tests, and notebooks. I installed it with `pip install -e .` so notebooks can import it directly.

I ran into a few setup issues worth remembering:

- `python` isn't a command on macOS, only `python3`. Inside an activated venv, plain `python` works.
- `git push` failed with "Repository not found" because `git remote add` only saves a URL. I had to create the empty repo on GitHub first, without a README or .gitignore so it wouldn't conflict with my first commit.
- The notebook kept picking the wrong Python (Homebrew 3.13, then a uv-managed 3.12), so it couldn't find ipykernel or my packages. The root cause was that VS Code was opened on my parent `Projects` folder, so it never looked inside `phishing-detector` for `.venv`. Opening the project folder directly fixed it. I also registered the venv as a named Jupyter kernel with `python -m ipykernel install --user --name phishing-detector`.
- `.pytest_cache` got committed, so I added it to `.gitignore`.

**Decision:** Always open the project folder itself in VS Code, and always check the kernel name in the top right before running cells.

---

## 2026-10-22 · Notebook 01: baseline and the scores were too good

**Goal:** Train simple models on 19 hand-built URL features and measure precision and recall on the phishing class.

**Data:** PhiUSIIL from UCI. After removing duplicate URLs, there are 235,370 URLs: 134,850 legit (57.3%) and 100,520 phishing (42.7%). The classes are balanced enough that accuracy isn't totally misleading, but I'm reporting precision, recall, and PR-AUC anyway.

**Results (test set, 47,074 URLs):**

| model | precision | recall | F1 | PR-AUC |
|---|---|---|---|---|
| logistic regression | 0.9978 | 0.9812 | 0.9894 | 0.9973 |
| random forest | 0.9989 | 0.9915 | 0.9952 | 0.9982 |

The random forest missed 171 phishing URLs and raised 22 false alarms. At a threshold of 0.916 it reached 1.000 precision with 0.9875 recall.

**Why I don't trust these numbers:** Two leaks in how the dataset was collected let the model cheat.

1. **Scheme leak.** 100% of legit URLs start with `https://www.`, compared with only 2.4% of phishing URLs. A one-line rule on that prefix alone would be right about 99% of the time. `has_https` was the top random forest feature.
2. **Path leak.** Legit URLs are all bare homepages. Their max length is 58 characters with a standard deviation under 5, while phishing URLs go up to 6,097. So "has any path at all" basically means phishing. `path_len` was the second most important random forest feature and had by far the largest logistic regression weight (around +77).

The errors back this up. Every missed phishing URL was a bare HTTPS homepage like `https://www.cfg.me`, which looks exactly like the legit data. The false alarms were legit sites with long hyphenated or digit-heavy hostnames, like `christoph-graupner-gesellschaft.de` and `112112ww.com`.

The only signal that looked real was TLDs. Phishing leans toward `.app`, `.io`, `.ru`, `.top`, and `.dev`, while legit is mostly `.com`, `.org`, and country codes.

**Decision:** Treat these scores as an upper bound that mostly measures the leak. In real traffic, a model like this would probably flag huge numbers of normal links that have paths (product pages, docs, search results). Next I'll remove the leaks and see what score is left.

---

## 2026-10-22 · Notebook 02: removing the leaks

**Goal:** Find the honest score.

**Plan:** Use the same train/test split for every experiment so the comparison is fair.

- **A, original:** same setup as notebook 01, for reference.
- **B, no scheme:** strip `http(s)://` and `www.`, and drop `has_https`. This fixes the scheme leak but not the path leak.
- **C, host only:** use only the hostname, and drop all path and query features. Neither class has a path anymore, so this should be the honest baseline.
- **D, character n-grams:** TF-IDF on 3 to 5 character chunks of the hostname, plus logistic regression. This lets the model learn patterns like brand names and TLDs without me hand-writing features.

I'll also score a handful of real legit URLs with paths (Amazon, GitHub, Google Docs) to show model B failing on links the dataset never contained.

I added `normalize_url` and `host_only` helpers to `features.py`, with tests, so the API and browser extension will clean URLs exactly the same way later on.

**Results (same test set, 47,074 URLs):**

| experiment | precision | recall | PR-AUC |
|---|---|---|---|
| A · original | 0.9989 | 0.9915 | 0.9982 |
| B · no scheme | 0.9504 | 0.7296 | 0.9101 |
| C · host only | 0.8877 | 0.6357 | 0.8577 |
| D · host char n-grams | 0.9192 | 0.7800 | 0.9328 |

**What I learned:** Removing the leaks dropped PR-AUC from 0.998 to 0.858 and recall
from 99% to 64%, so most of notebook 01's score came from how the data was collected.
In C, feature importance is spread across entropy, TLD length, dots, subdomains, and
digits, with no single dominant feature, which is what a non-cheating model looks like.

The char n-gram model (D) beat my hand-built features on every test-set metric. Its
top phishing n-grams are Freenom's free TLDs (.cf, .ga, .ml, .gq) plus shared hosting
(ipfs, web.app). Legit leans on .gov, .edu, and country TLDs. It also learned that
"jp." mid-hostname means impersonation (amazon-co-jp.evil.cn) while ".jp" at the end
is legit. The concern is that D is largely a TLD classifier, some of that is dataset
bias in how legit sites were chosen, and Freenom stopped free registrations in 2023,
so it's partly learning old attack patterns. That's a drift risk to monitor later.

**Path check after normalizing:** 0% of legit URLs have a path vs 23.4% of phishing,
so stripping the scheme alone (experiment B) still leaves the path leak. Only the
host-only view (C) removes both.

**Side finding:** some phishing is hosted on shared platforms like Firebase
(`*.web.app`) and IPFS gateways (`cloudflare-ipfs.com/ipfs/...`). For IPFS, the
hostname belongs to a legitimate service and the malicious part lives in the path,
so a host-only model is blind to it. That's a known limitation of experiment C,
and a reason to collect legit URLs with paths later so the path can be used safely.

**Reality check:** I scored 8 real legit URLs (Amazon, GitHub, Google Docs, Gmail,
Wikipedia, LinkedIn, Stack Overflow, VCU) and 4 made-up phishing URLs. All three
models caught every phishing URL. False alarms on the legit links at a 0.5 threshold:

| model | legit falsely flagged | test-set PR-AUC |
|---|---|---|
| B · no scheme | 8 of 8 | 0.910 |
| C · host only | 3 of 8 | 0.858 |
| D · char n-grams | 5 of 8 | 0.933 |

B flagged every link with a path, which confirms the path leak. More surprising, D
had the best test-set score but the worst real-world behavior. The test set shares
the training data's biases, so it can't reveal them. Only outside data can.

**Third leak, subdomains:** Every legit training URL was www.something.tld, so after
stripping www., legit hosts never have a subdomain. The models learned that any
subdomain means phishing, which is why docs.google.com, mail.google.com, and
en.wikipedia.org got flagged. Brand names like "amazon" and "linkedin" also look
phishy because they mostly appear in the phishing data as impersonation targets.

**Decision:** Model choice isn't the bottleneck. The legit data is. Next I'll build
a real-world evaluation set of legit URLs with subdomains and paths plus fresh
phishing URLs, and add better legit training data (e.g., the Tranco top domains
list). The saved host_model.joblib is a record, not a production model.

---

## 2026-10-22 · Notebook 03: real-world benchmark

**Goal:** Score every model on data from outside PhiUSIIL, since notebook 02 showed
the test set shares the training data's blind spots.

**Benchmark:** 5,000 Tranco domains (half top 10k, half long tail), 374 URLs from my
own Chrome history, and 300 live OpenPhish URLs from 2026-10-22. Under 8% of
benchmark hosts appeared in training, so results reflect generalization, not memory.

**Results (threshold 0.5):**

| model | Tranco FA | history FA | OpenPhish recall | bench PR-AUC |
|---|---|---|---|---|
| A · original | 0.910 | 1.000 | 1.000 | 0.360 |
| B · no scheme | 0.073 | 1.000 | 1.000 | 0.433 |
| C · host only | 0.119 | 0.390 | 0.880 | 0.737 |
| D · host char n-grams | 0.251 | 0.281 | 0.953 | 0.410 |

**What I learned:**

C beats D on real data (0.737 vs 0.410 PR-AUC), the reverse of the test-set ranking.
That's the second time the test set picked the wrong model. B's perfect recall is
meaningless because it flags anything with a path.

No model is usable yet. At 0.5, D raises about 1,360 false alarms to catch 286
phishing URLs, and real-world phishing is much rarer than in this benchmark.

The subdomain theory held. On my history, C flags 2% of legit URLs without a
subdomain and 84% of those with one.

Phishing has moved to free hosting platforms: 63% of today's OpenPhish URLs vs 23%
in training, while Freenom TLDs fell from 4.0% to 0.7%. The phishing part is an
attacker-chosen subdomain (e.g. something.webflow.io). So the "subdomain means
phishing" shortcut that causes false alarms on my history is also a big part of the
models' recall. Fixing one will probably cost some of the other.

**Benchmark caveats:** Tranco ranks popularity, not safety. Some of D's Tranco
"false alarms" are gambling mirrors, proxies, and download sites from the long tail,
so the Tranco FA rate is partly label noise. Host-only models also can't catch
compromised legit sites where the phishing lives in the path (a hacked winery site,
a WordPress install).

**Decision:** Notebook 04 fixes the training data and the features instead of
trying new models:
- Use the Public Suffix List (tldextract) to split hosts into subdomain, registered
  domain, and suffix properly. It also knows that workers.dev and webflow.io are
  shared hosting, where the subdomain belongs to whoever signed up.
- Add a "hosted on a free platform" feature instead of letting the model learn
  "subdomain means phishing."
- Add legit hostnames that have subdomains to training (from a DNS-based top list
  like Cisco Umbrella, which lists full hostnames like docs.google.com).
- Score Tranco by rank bucket so long-tail label noise is visible.