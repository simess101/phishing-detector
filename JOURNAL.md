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

## 2026-10-22 · Notebook 02: removing the leaks (plan)

**Goal:** Find the honest score.

**Plan:** Use the same train/test split for every experiment so the comparison is fair.

- **A, original:** same setup as notebook 01, for reference.
- **B, no scheme:** strip `http(s)://` and `www.`, and drop `has_https`. This fixes the scheme leak but not the path leak.
- **C, host only:** use only the hostname, and drop all path and query features. Neither class has a path anymore, so this should be the honest baseline.
- **D, character n-grams:** TF-IDF on 3 to 5 character chunks of the hostname, plus logistic regression. This lets the model learn patterns like brand names and TLDs without me hand-writing features.

I'll also score a handful of real legit URLs with paths (Amazon, GitHub, Google Docs) to show model B failing on links the dataset never contained.

I added `normalize_url` and `host_only` helpers to `features.py`, with tests, so the API and browser extension will clean URLs exactly the same way later on.

**Results:** _fill in after running_

| experiment | precision | recall | PR-AUC |
|---|---|---|---|
| A · original | | | |
| B · no scheme | | | |
| C · host only | | | |
| D · host char n-grams | | | |

**What I learned:** _fill in_
