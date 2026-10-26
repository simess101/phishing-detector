"""Character-level CNN on full URLs (stage 2).

Idea: read the normalized URL one character at a time, and let convolution
filters learn which short character patterns (3, 5, 7 characters wide) signal
phishing: "wp-includes/", ".php?", "-verify", "paypa1", random-looking hashes.
Unlike v1, it sees the path, so it can flag phishing planted on hacked sites.

    url -> normalize_full_url -> character ids -> embedding
        -> 3 parallel Conv1d (widths 3, 5, 7) -> max over positions
        -> small MLP -> P(phishing)

CharCNNClassifier has the same phish_proba(urls) interface as the v1 models,
so it can be dropped into PhishDetector, and it pickles to CPU tensors so
joblib can save it inside a detector.
"""

from __future__ import annotations

import time
from collections import Counter

import numpy as np

from .fullurl import MAX_LEN, normalize_full_url

PAD, UNK = 0, 1


def best_device():
    import torch
    if torch.backends.mps.is_available():   # Apple Silicon GPU
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class CharVocab:
    """Maps characters to integer ids. 0 = padding, 1 = unknown character."""

    def __init__(self, chars: list[str]):
        self.chars = list(chars)
        self.index = {c: i + 2 for i, c in enumerate(self.chars)}

    @classmethod
    def build(cls, texts, min_count: int = 5) -> "CharVocab":
        counts = Counter()
        for t in texts:
            counts.update(t)
        return cls(sorted(c for c, n in counts.items() if n >= min_count))

    def __len__(self):
        return len(self.chars) + 2

    def encode(self, texts, max_len: int = MAX_LEN) -> np.ndarray:
        out = np.zeros((len(texts), max_len), dtype=np.int64)
        for i, t in enumerate(texts):
            ids = [self.index.get(c, UNK) for c in t[:max_len]]
            out[i, :len(ids)] = ids
        return out


def _build_net(vocab_size: int, embed_dim: int = 48, filters: int = 128,
               widths=(3, 5, 7), dropout: float = 0.3):
    import torch
    from torch import nn

    class CharCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
            self.convs = nn.ModuleList([nn.Conv1d(embed_dim, filters, w, padding=w // 2) for w in widths])
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(filters * len(widths), 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )

        def forward(self, x):                       # x: (batch, length) of character ids
            mask = (x != PAD).unsqueeze(1)          # (batch, 1, length): ignore padding when pooling
            e = self.embed(x).transpose(1, 2)       # (batch, embed_dim, length)
            pooled = []
            for conv in self.convs:
                h = torch.relu(conv(e))[..., : x.shape[1]]
                # strongest match anywhere in the URL. ReLU output is >= 0, so zeroing the
                # padding positions keeps them from ever winning the max.
                pooled.append(h.masked_fill(~mask, 0.0).amax(dim=2))
            return self.head(torch.cat(pooled, dim=1)).squeeze(1)      # one logit per URL

    return CharCNN()


class CharCNNClassifier:
    """Trainable wrapper with the phish_proba(urls) interface used everywhere else."""

    def __init__(self, embed_dim=48, filters=128, widths=(3, 5, 7), dropout=0.3,
                 max_len=MAX_LEN, min_char_count=5, seed=42, drop_query=False):
        self.config = dict(embed_dim=embed_dim, filters=filters, widths=tuple(widths), dropout=dropout)
        self.max_len, self.min_char_count, self.seed = max_len, min_char_count, seed
        self.drop_query = drop_query
        self.vocab, self.net, self.history = None, None, []
        self.device = None

    def _norm(self, url):
        # getattr: models saved before drop_query existed (notebook 07) still load
        return normalize_full_url(url, self.max_len, drop_query=getattr(self, "drop_query", False))

    # ----- training -------------------------------------------------------
    def fit(self, urls, y, val_urls=None, val_y=None, epochs=10, batch_size=256, lr=2e-3,
            weight_decay=1e-4, patience=2, device=None, verbose=True, max_batches=None):
        """Train with AdamW. If validation data is given, keep the epoch with the best
        validation PR-AUC and stop after `patience` epochs without improvement.
        max_batches limits batches per epoch (for quick smoke tests)."""
        import torch
        from sklearn.metrics import average_precision_score

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.device = device or best_device()

        texts = [self._norm(u) for u in urls]
        self.vocab = CharVocab.build(texts, self.min_char_count)
        X = torch.from_numpy(self.vocab.encode(texts, self.max_len))
        Y = torch.tensor(np.asarray(y, dtype=np.float32))     # copy: avoids a read-only warning

        self.net = _build_net(len(self.vocab), **self.config).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        best, best_state, bad = -1.0, None, 0
        self.history = []
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            self.net.train()
            order = rng.permutation(len(X))
            total, n = 0.0, 0
            for b, start in enumerate(range(0, len(order), batch_size)):
                if max_batches and b >= max_batches:
                    break
                idx = torch.from_numpy(order[start:start + batch_size])
                xb, yb = X[idx].to(self.device), Y[idx].to(self.device)
                opt.zero_grad()
                loss = loss_fn(self.net(xb), yb)
                loss.backward()
                opt.step()
                total += loss.item() * len(idx); n += len(idx)

            row = {"epoch": epoch, "train_loss": total / max(n, 1), "seconds": time.time() - t0}
            if val_urls is not None:
                p = self.phish_proba(val_urls)
                row["val_pr_auc"] = average_precision_score(val_y, p)
                if row["val_pr_auc"] > best:
                    best, bad = row["val_pr_auc"], 0
                    best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                else:
                    bad += 1
            self.history.append(row)
            if verbose:
                extra = f"  val PR-AUC {row['val_pr_auc']:.4f}" if "val_pr_auc" in row else ""
                print(f"epoch {epoch:2d}  loss {row['train_loss']:.4f}{extra}  ({row['seconds']:.0f}s)", flush=True)
            if val_urls is not None and bad >= patience:
                if verbose:
                    print(f"no improvement for {patience} epochs, stopping")
                break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    # ----- scoring --------------------------------------------------------
    def phish_proba(self, urls, batch_size=1024) -> np.ndarray:
        import torch
        if self.net is None:
            raise RuntimeError("model is not trained")
        device = self.device or best_device()
        self.net.to(device).eval()
        texts = [self._norm(u) for u in urls]
        X = torch.from_numpy(self.vocab.encode(texts, self.max_len))
        out = []
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                out.append(torch.sigmoid(self.net(X[start:start + batch_size].to(device))).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    # ----- saving (joblib / pickle) -----------------------------------------
    def __getstate__(self):
        state = self.__dict__.copy()
        if self.net is not None:
            state["net"] = {k: v.detach().cpu() for k, v in self.net.state_dict().items()}
        state["device"] = None          # pick the best device again after loading
        return state

    def __setstate__(self, state):
        weights = state.get("net")
        self.__dict__.update(state)
        if weights is not None:
            self.net = _build_net(len(self.vocab), **self.config)
            self.net.load_state_dict(weights)
            self.net.eval()
