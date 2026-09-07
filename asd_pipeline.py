#!/usr/bin/env python3
"""
Chaine ASD complete pour DCASE 2026 Task 2, deux canaux.

Deux front-ends interchangeables (--frontend) :
    cnn   petit CNN discriminatif entraine de zero sur les donnees du challenge
          (classification type de machine x attributs, perte ArcFace)
    ast   Audio Spectrogram Transformer pre-entraine et GELE, aucun entrainement
          -> c'est la ligne de reference publiee a 60.28 sur le dev set

Dans les deux cas le score d'anomalie est la distance cosinus au plus proche
voisin parmi les clips normaux d'entrainement.

Contribution testee : un masque calcule a partir des deux canaux, applique au
mel avant le modele. Aucune donnee externe, aucun poids pre-entraine.

Variantes de masque (--mask) :
    none        canal 1 seul, sans masque            -> reference S3
    leveldiff   ecart de niveau proche/lointain      -> contribution S4
    coherence   coherence magnitude-quadratique      -> variante S4b

Usage :
    python asd_pipeline.py --data-root ./dev_data --frontend ast --mask none
    python asd_pipeline.py --data-root ./dev_data --frontend ast --mask leveldiff
    python asd_pipeline.py --data-root ./dev_data --frontend cnn --mask none      --seed 0
    python asd_pipeline.py --data-root ./dev_data --frontend cnn --mask leveldiff --seed 0

Le front-end ast n'a pas de seed a repeter : il ne s'entraine pas, il est
deterministe. Il telecharge son checkpoint au premier appel (~350 Mo).

Les features sont mises en cache par (machine, frontend, mask, alpha, beta) :
les variantes suivantes ne recalculent rien.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import wavfile
from scipy.signal import coherence as sp_coherence
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

SPLITS = ("train", "test", "supplemental")
N_FFT, HOP, N_MELS, FMIN = 1024, 512, 128, 20.0
EPS = 1e-10


# ==========================================================================
# Donnees
# ==========================================================================

def discover(data_root):
    """{machine: {split: [paths]}}, a n'importe quelle profondeur."""
    found = {}
    for dirpath, dirnames, _ in os.walk(data_root):
        d = Path(dirpath)
        if d.name in SPLITS:
            continue
        splits = {}
        for s in SPLITS:
            if s in dirnames:
                w = sorted((d / s).glob("*.wav"))
                if w:
                    splits[s] = w
        if splits:
            found[d.name] = splits
            dirnames[:] = [x for x in dirnames if x not in SPLITS]
    return dict(sorted(found.items()))


def parse_name(path):
    """
    section_00_source_test_anomaly_0012_car_A1_spd_28V.wav
    -> domaine, label, chaine d'attributs (classe pour l'entrainement).
    """
    stem = path.stem
    dom = "source" if "_source_" in stem else ("target" if "_target_" in stem else "unknown")
    lab = "anomaly" if "_anomaly_" in stem else ("normal" if "_normal_" in stem else "unknown")
    m = re.search(r"_(?:normal|anomaly)_\d+_?(.*)$", stem)
    attr = m.group(1) if m and m.group(1) else "noAttributes"
    return dom, lab, attr


# ==========================================================================
# Features
# ==========================================================================

def mel_filterbank(sr, n_fft, n_mels, fmin):
    """Banc de filtres mel triangulaires, sans dependance a librosa."""
    fmax = sr / 2
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)
    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        if c > l:
            fb[i, l:c] = (np.arange(l, c) - l) / (c - l)
        if r > c:
            fb[i, c:r] = (r - np.arange(c, r)) / (r - c)
    return fb


def stft_power(x, n_fft, hop):
    win = np.hanning(n_fft + 1)[:-1]
    n_frames = 1 + (len(x) - n_fft) // hop
    if n_frames < 1:
        x = np.pad(x, (0, n_fft - len(x)))
        n_frames = 1
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx] * win
    return np.abs(np.fft.rfft(frames, axis=1)).T ** 2  # (F, T)


# AST attend un fbank kaldi 128 mels, fenetre 25 ms / pas 10 ms, et sa propre
# normalisation. On calcule donc les deux canaux sur CETTE grille pour que le
# masque s'applique au bon endroit.
AST_MEAN, AST_STD, AST_FRAMES = -4.2677393, 4.5689974, 1024


def kaldi_mel_power(x, sr):
    """Energies mel lineaires (128, T) sur la grille attendue par AST."""
    import torchaudio
    w = torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0)
    fb = torchaudio.compliance.kaldi.fbank(
        w, htk_compat=True, sample_frequency=sr, use_energy=False,
        window_type="hanning", num_mel_bins=128, dither=0.0,
        frame_shift=10, use_log_fbank=False)
    return fb.numpy().T


def read_stereo(path):
    sr, d = wavfile.read(path)
    if d.ndim != 2 or d.shape[1] < 2:
        raise ValueError(f"{path.name}: 2 canaux attendus, recu shape={d.shape}")
    if np.issubdtype(d.dtype, np.integer):
        d = d.astype(np.float64) / np.iinfo(d.dtype).max
    return d[:, 0], d[:, 1], sr


def compute_feature(path, fb, mask, alpha, beta, frontend="cnn", gfloor=0.05,
                    calib_pct=20.0, swap=False, want_noise=False):
    """
    Retourne un log-mel (N_MELS, T) eventuellement masque.

    leveldiff : le rapport de puissance entre les deux canaux vaut 1+SNR, mais
    seulement une fois corrige de l'ecart de gain entre les deux microphones.
    Sans cette calibration le masque sature (plancher ou 1 partout) et devient
    inerte. On estime donc le facteur c sur un percentile bas du rapport P1/P2,
    c'est-a-dire dans les bandes ou la machine est absente, puis :

        G = 1 - beta * c / (P1/P2)

    Dans les bandes de bruit le rapport vaut c, donc G tombe au plancher. Dans
    les bandes ou la machine emet, le rapport depasse c et G remonte vers 1.
    Aucune etiquette, aucune mesure prealable : tout est estime par clip.

    coherence : gain = MSC(f) par bin, projetee sur l'echelle mel. Une source
    ponctuelle fixe donne une MSC proche de 1, un champ diffus s'effondre.
    """
    x1, x2, sr = read_stereo(path)
    if swap:
        x1, x2 = x2, x1
    n = min(len(x1), len(x2))
    x1, x2 = x1[:n], x2[:n]

    if frontend == "ast":
        M1 = kaldi_mel_power(x1, sr)
        if mask == "none":
            g = 1.0
        elif mask == "leveldiff":
            M2 = kaldi_mel_power(x2, sr)
            T = min(M1.shape[1], M2.shape[1]); M1, M2 = M1[:, :T], M2[:, :T]
            g = wiener_gain(M1, M2, alpha, beta, gfloor, calib_pct)
        elif mask == "coherence":
            f, msc = sp_coherence(x1, x2, fs=sr, nperseg=N_FFT, noverlap=N_FFT // 2)
            fbn = fb / (fb.sum(axis=1, keepdims=True) + EPS)
            g = np.clip(fbn @ msc, gfloor, 1.0)[:, None] ** alpha
        else:
            raise ValueError(f"masque inconnu : {mask}")
        out = ((np.log(M1 * g + EPS) - AST_MEAN) / (AST_STD * 2)).astype(np.float32)
        if not want_noise:
            return out
        M2f = kaldi_mel_power(x2, sr)[:, :M1.shape[1]]
        nz = ((np.log(M2f + EPS) - AST_MEAN) / (AST_STD * 2)).astype(np.float32)
        return out, nz

    P1 = stft_power(x1, N_FFT, HOP)

    if mask == "none":
        M1 = fb @ P1
        out = np.log(M1 + EPS).astype(np.float32)
        if not want_noise:
            return out
        return out, np.log(fb @ stft_power(x2, N_FFT, HOP) + EPS).astype(np.float32)

    if mask == "leveldiff":
        P2 = stft_power(x2, N_FFT, HOP)
        M1, M2 = fb @ P1, fb @ P2
        g = wiener_gain(M1, M2, alpha, beta, gfloor, calib_pct)
        return np.log(M1 * g + EPS).astype(np.float32)

    if mask == "coherence":
        f, msc = sp_coherence(x1, x2, fs=sr, nperseg=N_FFT, noverlap=N_FFT // 2)
        w = fb / (fb.sum(axis=1, keepdims=True) + EPS)
        g = np.clip(w @ msc, gfloor, 1.0)[:, None] ** alpha
        M1 = fb @ P1
        return np.log(M1 * g + EPS).astype(np.float32)

    raise ValueError(f"masque inconnu : {mask}")


def wiener_gain(M1, M2, alpha, beta, gfloor, calib_pct):
    """Gain de Wiener calibre sur l'ecart de gain entre les deux microphones."""
    ratio = M1 / (M2 + EPS)
    c = np.percentile(ratio, calib_pct)
    return np.clip(1.0 - beta * c / (ratio + EPS), gfloor, 1.0) ** alpha


def _denorm(x, frontend):
    """Retour au log-mel brut : le front-end AST stocke une version normalisee."""
    return x * (AST_STD * 2) + AST_MEAN if frontend == "ast" else x


def _renorm(logm, frontend):
    if frontend == "ast":
        return ((logm - AST_MEAN) / (AST_STD * 2)).astype(np.float32)
    return logm.astype(np.float32)


def farmix_round(X, N, snr_lo, snr_hi, rng, frontend, chunk=200):
    """
    Une copie augmentee de la banque de reference.

    Chaque clip normal recoit du bruit preleve sur le CANAL 2 d'un AUTRE clip,
    a un rapport signal sur bruit tire au hasard. Le canal 2 est domine par le
    bruit d'usine reel, enregistre dans la meme salle avec le meme materiel :
    c'est une banque de bruit en domaine, disponible gratuitement.

    Le melange se fait dans le domaine mel-puissance, ou les puissances de deux
    signaux decorreles s'additionnent. On evite ainsi de relire les fichiers
    audio a chaque variante.

    Le decalage circulaire garantit qu'aucun clip ne recoit son propre canal 2,
    ce qui reinjecterait la signature de la machine au lieu du bruit.
    """
    n = len(X)
    perm = (np.arange(n) + int(rng.integers(1, max(2, n)))) % n
    out = np.empty_like(X)
    for i in range(0, n, chunk):
        sl = slice(i, min(i + chunk, n))
        M1 = np.exp(_denorm(X[sl], frontend).astype(np.float32))
        N2 = np.exp(_denorm(N[perm[sl]], frontend).astype(np.float32))
        snr = rng.uniform(snr_lo, snr_hi, size=(M1.shape[0], 1, 1))
        lam = (M1.sum(axis=(1, 2), keepdims=True)
               / (10 ** (snr / 10) * N2.sum(axis=(1, 2), keepdims=True) + EPS))
        out[sl] = _renorm(np.log(M1 + lam * N2 + EPS), frontend)
    return out


def cache_key(machine, frontend, mask, alpha, beta, gfloor, extra=""):
    h = hashlib.md5(f"{frontend}|{mask}|{alpha}|{beta}|{gfloor}|{extra}".encode()).hexdigest()[:8]
    return f"{machine}__{frontend}_{mask}_{h}.npz"


def build_features(machine, splits, args, fb, cache_dir):
    """Calcule (ou recharge) les features de train et test d'une machine."""
    need_noise = args.farmix > 0
    ck = cache_dir / cache_key(machine, args.frontend, args.mask, args.alpha,
                               args.beta, args.gfloor,
                               f"{args.calib_pct}|{args.swap_channels}|fm{int(args.farmix > 0)}")
    if ck.exists() and not args.no_cache:
        z = np.load(ck, allow_pickle=True)
        d = {k: z[k] for k in z.files}
        if "test_file" in d or "train_file" in d:
            return d
        print(f"  cache de {machine} anterieur aux noms de fichiers, recalcul")

    items = {"train": [], "test": []}
    for split in ("train", "test"):
        for p in splits.get(split, []):
            dom, lab, attr = parse_name(p)
            try:
                r = compute_feature(p, fb, args.mask, args.alpha, args.beta,
                                    args.frontend, args.gfloor, args.calib_pct,
                                    args.swap_channels, want_noise=need_noise)
                x, nz = r if need_noise else (r, None)
            except Exception as e:
                print(f"  ! {p.name}: {e}", file=sys.stderr)
                continue
            items[split].append((x, dom, lab, attr, p.name, nz))

    out = {}
    for split in ("train", "test"):
        if not items[split]:
            continue
        T = min(x.shape[1] for x, *_ in items[split])
        out[f"{split}_X"] = np.stack([x[:, :T] for x, *_ in items[split]])
        out[f"{split}_dom"] = np.array([d for _, d, _, _, _, _ in items[split]])
        out[f"{split}_lab"] = np.array([l for _, _, l, _, _, _ in items[split]])
        out[f"{split}_attr"] = np.array([a for _, _, _, a, _, _ in items[split]])
        out[f"{split}_file"] = np.array([f for _, _, _, _, f, _ in items[split]])
        if need_noise and split == "train":
            out["train_N"] = np.stack([z[:, :T] for *_, z in items[split]])
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ck, **out)
    return out


# ==========================================================================
# Modele
# ==========================================================================

class ArcFace(nn.Module):
    """Marge angulaire additive. Resserre les classes, ce que le kNN exploite."""

    def __init__(self, dim, n_classes, scale=30.0, margin=0.7):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_classes, dim) * 0.01)
        self.scale, self.margin = scale, margin

    def forward(self, emb, labels):
        cos = F.linear(F.normalize(emb), F.normalize(self.W)).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos)
        one_hot = F.one_hot(labels, self.W.shape[0]).float()
        return self.scale * torch.cos(theta + self.margin * one_hot)


class SmallCNN(nn.Module):
    """~1.1 M parametres. Entierement convolutif : accepte une duree variable."""

    def __init__(self, emb_dim=128, n_classes=10):
        super().__init__()
        ch = [1, 32, 64, 128, 256]
        blocks = []
        for i in range(4):
            blocks += [nn.Conv2d(ch[i], ch[i + 1], 3, padding=1, bias=False),
                       nn.BatchNorm2d(ch[i + 1]), nn.ReLU(inplace=True),
                       nn.Conv2d(ch[i + 1], ch[i + 1], 3, padding=1, bias=False),
                       nn.BatchNorm2d(ch[i + 1]), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
        self.body = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, emb_dim)
        self.head = ArcFace(emb_dim, n_classes)

    def embed(self, x):
        h = self.pool(self.body(x.unsqueeze(1))).flatten(1)
        return self.fc(h)


# ==========================================================================
# Front-end gele
# ==========================================================================

class FrozenAST:
    """
    AST pre-entraine sur AudioSet, poids geles, aucun entrainement.

    On lui donne directement input_values (fbank normalise) plutot que de
    passer par son feature extractor : le masque a deja ete applique sur le
    mel, il ne faut pas le recalculer depuis la forme d'onde.
    """

    NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

    def __init__(self, device, pooling="mean"):
        from transformers import ASTModel
        self.model = ASTModel.from_pretrained(self.NAME).to(device).eval()
        for prm in self.model.parameters():
            prm.requires_grad_(False)
        self.device = device
        self.pooling = pooling
        self.n_params = sum(prm.numel() for prm in self.model.parameters())

        cfg = self.model.config
        self.f_dim = (cfg.num_mel_bins - cfg.patch_size) // cfg.frequency_stride + 1
        self.t_dim = (cfg.max_length - cfg.patch_size) // cfg.time_stride + 1
        self.n_special = 2   # jetons CLS et distillation, en tete de sequence
        self.n_bands = self.f_dim if pooling.startswith("freq") else 1
        if pooling.startswith("freq"):
            print(f"  grille de tokens : {self.f_dim} bandes x {self.t_dim} positions")

    # Valeur de remplissage : AST complete le fbank BRUT avec des zeros puis
    # normalise. On reproduit exactement ce comportement.
    PAD = (0.0 - AST_MEAN) / (AST_STD * 2)

    @classmethod
    def _windows(cls, x):
        """
        Decoupe en fenetres de AST_FRAMES trames. Les clips du challenge font
        10 a 12 s, soit 1000 a 1200 trames : tronquer perdrait la fin des plus
        longs, on moyenne donc les embeddings des fenetres.
        """
        T = x.shape[-1]
        n_win = max(1, int(np.ceil(T / AST_FRAMES)))
        out = []
        for k in range(n_win):
            seg = x[:, k * AST_FRAMES:(k + 1) * AST_FRAMES]
            if seg.shape[1] < AST_FRAMES:
                seg = np.pad(seg, ((0, 0), (0, AST_FRAMES - seg.shape[1])),
                             mode="constant", constant_values=cls.PAD)
            out.append(seg)
        return out

    @torch.no_grad()
    def embed(self, X, batch=8):
        """
        Retourne (N, n_bandes, d).

        Les tokens d'AST forment une grille bande de frequence x position
        temporelle. Les moyenner tous ecrase l'axe frequentiel, alors que la
        signature d'un defaut mecanique vit dans une bande precise. Selon
        --pooling on garde donc les bandes separees.
        """
        wins, owner = [], []
        for i, x in enumerate(X):
            w = self._windows(x)
            wins += w
            owner += [i] * len(w)
        owner = np.array(owner)

        emb = []
        for i in range(0, len(wins), batch):
            xb = np.stack(wins[i:i + batch])
            # (B, mels, T) -> (B, T, mels), la convention d'AST
            t = torch.from_numpy(xb).float().transpose(1, 2).to(self.device)
            h = self.model(input_values=t).last_hidden_state      # (B, n_tok, d)

            if self.pooling == "mean":
                v = h.mean(dim=1, keepdim=True)                   # (B, 1, d)
            elif self.pooling == "meanstd":
                v = torch.cat([h.mean(1), h.std(1)], dim=-1)[:, None, :]
            else:
                g = h[:, self.n_special:, :]                      # tokens de patch
                B, _, d = g.shape
                g = g.reshape(B, self.f_dim, self.t_dim, d)       # (B, freq, temps, d)
                if self.pooling == "freq":
                    v = g.mean(dim=2)                             # (B, freq, d)
                else:                                             # freq_meanstd
                    v = torch.cat([g.mean(dim=2), g.std(dim=2)], dim=-1)
            emb.append(v.cpu().numpy())

        emb = np.concatenate(emb)                                 # (n_win, bandes, d)
        return np.stack([emb[owner == i].mean(0) for i in range(len(X))])


# ==========================================================================
# Entrainement
# ==========================================================================

def train_model(Xtr, ytr, n_classes, args, device):
    """
    Un seul modele pour toutes les machines : la classification inter-machines
    fournit le signal discriminant (outlier exposure implicite).
    """
    model = SmallCNN(args.emb_dim, n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    n, crop = len(Xtr), args.crop
    ytr_t = torch.from_numpy(ytr).long()

    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        tot = corr = 0
        loss_sum = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb = []
            for j in idx.tolist():
                T = Xtr[j].shape[1]
                s = np.random.randint(0, max(1, T - crop + 1))
                seg = Xtr[j][:, s:s + crop]
                if seg.shape[1] < crop:
                    seg = np.pad(seg, ((0, 0), (0, crop - seg.shape[1])), mode="wrap")
                xb.append(seg)
            xb = torch.from_numpy(np.stack(xb)).float().to(device)
            yb = ytr_t[idx].to(device)
            xb = (xb - xb.mean(dim=(1, 2), keepdim=True)) / (xb.std(dim=(1, 2), keepdim=True) + 1e-5)

            # SpecAugment : masque des bandes de frequence et des segments
            # temporels. Sans lui le reseau memorise les clips, l'accuracy
            # atteint 1.0 en quelques epoques et les embeddings s'effondrent
            # sur les centres de classe, ce qui detruit la detection.
            if args.specaug > 0:
                B, Fm, Tm = xb.shape
                for _ in range(2):
                    w = int(np.random.randint(0, int(args.specaug * Fm) + 1))
                    if w:
                        f0 = int(np.random.randint(0, Fm - w + 1))
                        xb[:, f0:f0 + w, :] = 0
                    w = int(np.random.randint(0, int(args.specaug * Tm) + 1))
                    if w:
                        t0 = int(np.random.randint(0, Tm - w + 1))
                        xb[:, :, t0:t0 + w] = 0

            logits = model.head(model.embed(xb), yb)
            loss = F.cross_entropy(logits, yb, label_smoothing=args.label_smoothing)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * len(idx)
            corr += (logits.argmax(1) == yb).sum().item(); tot += len(idx)
        sched.step()
        if (ep + 1) % max(1, args.epochs // 5) == 0 or ep == 0:
            print(f"    epoch {ep+1:3d}/{args.epochs}  loss {loss_sum/tot:.4f}  acc {corr/tot:.3f}")
    return model


@torch.no_grad()
def embed_all(model, X, device, batch=32):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(np.stack(X[i:i + batch])).float().to(device)
        xb = (xb - xb.mean(dim=(1, 2), keepdim=True)) / (xb.std(dim=(1, 2), keepdim=True) + 1e-5)
        out.append(F.normalize(model.embed(xb)).cpu().numpy()[:, None, :])
    return np.concatenate(out)


# ==========================================================================
# Score et metriques
# ==========================================================================

def _l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + EPS)


def _band_knn(E, Q, k, skip_self=False):
    """
    Distance cosinus moyenne aux k plus proches voisins, bande par bande.

    skip_self : quand Q est la banque elle-meme, la distance la plus petite est
    celle du clip a lui-meme, donc nulle. Il faut l'ecarter et non la moyenner,
    sinon le score de reference est sous-estime et tout seuil calibre dessus est
    trop bas.
    """
    off = 1 if skip_self else 0
    per_band = []
    for b in range(E.shape[1]):
        d = 1.0 - Q[:, b, :] @ E[:, b, :].T
        kk = max(1, min(k, d.shape[1] - off))
        per_band.append(np.sort(d, axis=1)[:, off:off + kk].mean(axis=1))
    return np.stack(per_band, axis=1)                  # (n_test, n_bandes)


def knn_scores(E_train, dom_train, E_test, k, align, agg="mean", per_domain=False,
               skip_self=False):
    """
    Score d'anomalie par plus proches voisins parmi les clips normaux.

    per_domain : cherche le voisin separement dans la banque source et dans la
    banque cible, puis retient la plus petite des deux distances.

    Sans cette separation, un clip du domaine cible est presque toujours
    apparie a une reference source, puisqu'il y en a 990 contre 10. La distance
    mesuree reflete alors l'ecart entre domaines et non la presence d'une
    anomalie : plus un clip est typiquement "cible", plus il parait anormal.
    C'est ce qui produit les AUC inversees observees sur le jeu d'evaluation
    (ToyDrone : 79.5 en source, 27.8 en cible).

    Aucune etiquette de domaine des clips de TEST n'est utilisee : seule la
    banque d'entrainement est partitionnee. La methode reste donc applicable a
    l'evaluation, ou le domaine des clips de test est inconnu.

    align : recentre les references du domaine cible sur la moyenne du domaine
    source. Incompatible avec per_domain, qui traite le probleme autrement.

    Les embeddings arrivent en (N, n_bandes, d). La recherche est faite bande
    par bande, puis les distances sont agregees : une anomalie confinee a une
    bande n'est plus diluee par les bandes intactes.
    """
    E = _l2(E_train.copy())
    Q = _l2(E_test)

    if per_domain:
        src, tgt = dom_train == "source", dom_train == "target"
        banks = [E[m] for m in (src, tgt) if m.sum() > 0]
        if len(banks) > 1:
            # Prendre le minimum brut entre les deux banques ne servirait a rien :
            # le minimum sur une union vaut le minimum des minimums. Il faut
            # normaliser chaque banque par sa propre echelle. Avec 990 references
            # source contre 10 cible, le plus proche voisin est mecaniquement plus
            # loin cote cible, et la banque source gagne toujours.
            # L'echelle est la distance typique des references d'une banque entre
            # elles (laisse-un-de-cote), donc estimee sans aucun clip de test.
            Ds = []
            for b in banks:
                self_d = _band_knn(b, b, k, skip_self=True)
                scale = np.median(self_d, axis=0, keepdims=True) + EPS
                Ds.append(_band_knn(b, Q, k, skip_self) / scale)
            D = np.minimum.reduce(Ds)
        else:
            D = _band_knn(E, Q, k, skip_self)
    else:
        if align:
            src, tgt = dom_train == "source", dom_train == "target"
            if src.any() and tgt.any():
                E[tgt] += E[src].mean(0) - E[tgt].mean(0)
                E = _l2(E)
        D = _band_knn(E, Q, k, skip_self)

    # mean : dilue une anomalie confinee mais reste robuste au bruit.
    # max  : sensible aux anomalies tres localisees, plus bruite.
    return D.max(axis=1) if agg == "max" else D.mean(axis=1)


def choose_threshold(train_scores, test_scores, method, q):
    """
    Fixe le seuil normal/anomalie sans jamais voir de clip anormal etiquete.

    gamma      loi gamma ajustee sur les scores des clips normaux d'entrainement,
               quantile q. C'est la procedure de la baseline officielle.
    percentile quantile q empirique des memes scores, sans hypothese de forme.
    mixture    melange de deux gaussiennes ajuste sur les scores de TEST en
               echelle log, seuil au croisement des deux composantes. Suppose
               que le lot de test contient les deux populations.
    otsu       seuil maximisant la variance inter-classes des scores de test,
               methode classique de seuillage d'histogramme.

    Les deux dernieres exploitent la distribution des scores de test, donc un
    acces par lot aux donnees. Aucune etiquette n'est utilisee.
    """
    if method == "percentile":
        return float(np.percentile(train_scores, q * 100))

    if method == "gamma":
        from scipy.stats import gamma as gamma_dist
        # Avec --per-domain les scores sont resserres autour de 1 et strictement
        # positifs. Fixer floc au minimum observe ne laisse aucune marge et
        # l'ajustement echoue : on essaie donc floc=0 en premier.
        for floc in (0.0, float(np.min(train_scores)) - 1e-6):
            try:
                sh, loc, sc = gamma_dist.fit(train_scores, floc=floc)
                t = float(gamma_dist.ppf(q, sh, loc=loc, scale=sc))
                if np.isfinite(t):
                    return t
            except Exception:
                continue
        print("    [gamma] ajustement echoue, repli sur le percentile empirique")
        return float(np.percentile(train_scores, q * 100))

    v = np.log(np.asarray(test_scores) + EPS)

    if method == "otsu":
        # Forme robuste : les scores en echelle log sont negatifs, une
        # formulation par sommes ponderees normalisees changerait de signe.
        hist, edges = np.histogram(v, bins=128)
        centers = (edges[:-1] + edges[1:]) / 2
        w0 = hist.cumsum()
        w1 = hist.sum() - w0
        mu = (hist * centers).cumsum()
        mu_t = (hist * centers).sum()
        mean0 = mu / np.maximum(w0, 1)
        mean1 = (mu_t - mu) / np.maximum(w1, 1)
        inter = w0 * w1 * (mean0 - mean1) ** 2
        return float(np.exp(centers[int(np.argmax(inter))]))

    if method == "mixture":
        from sklearn.mixture import GaussianMixture
        gm = GaussianMixture(2, n_init=5, random_state=0).fit(v.reshape(-1, 1))
        lo, hi = np.sort(gm.means_.ravel())
        grid = np.linspace(lo, hi, 512).reshape(-1, 1)
        post = gm.predict_proba(grid)[:, int(np.argmax(gm.means_.ravel()))]
        return float(np.exp(grid[int(np.argmin(np.abs(post - 0.5)))][0]))

    raise ValueError(f"seuillage inconnu : {method}")


def pauc(y, s, p=0.1):
    """
    AUC partielle sur FPR <= p.

    C'est l'aire sous la courbe ROC restreinte aux faibles taux de faux
    positifs, standardisee (McClish) pour qu'un systeme aleatoire donne 0.5.
    On delegue a scikit-learn via max_fpr, ce qu'utilisent les baselines DCASE :
    reimplementer cette metrique est le moyen le plus sur de se tromper.
    """
    y, s = np.asarray(y), np.asarray(s)
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, s, max_fpr=p)


def machine_metrics(dom, lab, scores, thr=None):
    y = (lab == "anomaly").astype(int)
    out = {}
    for d in ("source", "target"):
        m = dom == d
        if m.sum() and len(np.unique(y[m])) == 2:
            out[f"AUC_{d}"] = roc_auc_score(y[m], scores[m]) * 100
    out["pAUC"] = pauc(y, scores) * 100
    if thr is not None:
        pred = (scores > thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        prec = tp / max(1, pred.sum())
        rec = tp / max(1, y.sum())
        out["prec"] = prec * 100
        out["rec"] = rec * 100
        out["F1"] = (2 * prec * rec / (prec + rec) * 100) if prec + rec else 0.0
    return out


def official_score(per_machine):
    """
    Moyenne harmonique des AUC et pAUC uniquement.

    Le filtrage sur les cles est indispensable : precision, rappel et F1 sont
    stockes dans le meme dictionnaire mais ne font pas partie du score officiel.
    """
    keep = ("AUC_source", "AUC_target", "pAUC")
    vals = [v for d in per_machine.values() for k, v in d.items()
            if k in keep and v == v]
    return len(vals) / np.sum(1.0 / np.maximum(np.array(vals), 1e-6)) if vals else float("nan")


def export_dcase_csv(machine, d, scores, thr, args, outdir):
    """
    Ecrit les deux CSV attendus par l'evaluateur officiel DCASE :
    anomaly_score_<machine>_section_00_test.csv  (nom de fichier, score)
    decision_result_<machine>_section_00_test.csv (nom de fichier, 0 ou 1)
    """
    sub = outdir / "dcase_submission"
    sub.mkdir(parents=True, exist_ok=True)
    names = d.get("test_file")
    if names is None:
        names = np.array([f"{i:04d}.wav" for i in range(len(scores))])
    sec = "section_00"
    with open(sub / f"anomaly_score_{machine}_{sec}_test.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerows([[n, f"{v:.10f}"] for n, v in zip(names, scores)])
    with open(sub / f"decision_result_{machine}_{sec}_test.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerows([[n, int(v > thr)] for n, v in zip(names, scores)])


# ==========================================================================
# Main
# ==========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True,
                    help="pointer sur le sous-dossier dev/, pas sur la racine "
                         "(la racine contient aussi eval/, non labellise)")
    ap.add_argument("--machines", default=None,
                    help="liste separee par des virgules pour restreindre l'analyse")
    ap.add_argument("--out", default="asd_results")
    ap.add_argument("--frontend", default="cnn", choices=["cnn", "ast"],
                    help="cnn = entraine de zero ; ast = pre-entraine et gele")
    ap.add_argument("--mask", default="none", choices=["none", "leveldiff", "coherence"])
    ap.add_argument("--alpha", type=float, default=1.0, help="exposant du masque")
    ap.add_argument("--beta", type=float, default=1.0, help="sur-soustraction (leveldiff)")
    ap.add_argument("--calib-pct", type=float, default=20.0,
                    help="percentile du rapport P1/P2 servant a calibrer l'ecart "
                         "de gain entre microphones")
    ap.add_argument("--swap-channels", action="store_true",
                    help="traite le canal 2 comme le micro proche")
    ap.add_argument("--gfloor", type=float, default=0.05,
                    help="plancher du gain : borne l'attenuation a -13 dB par defaut")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--crop", type=int, default=128, help="trames par crop d'entrainement")
    ap.add_argument("--specaug", type=float, default=0.15,
                    help="fraction max masquee par SpecAugment (0 = desactive)")
    ap.add_argument("--label-smoothing", type=float, default=0.1,
                    help="lissage des etiquettes, freine la memorisation")
    ap.add_argument("--emb-dim", type=int, default=128)
    ap.add_argument("--knn", type=int, default=1)
    ap.add_argument("--export-scores", action="store_true",
                    help="ecrit les CSV au format attendu par l'evaluateur officiel DCASE")
    ap.add_argument("--band-agg", default="mean", choices=["mean", "max"],
                    help="agregation des distances entre bandes (pooling freq)")
    ap.add_argument("--pooling", default="mean",
                    choices=["mean", "meanstd", "freq", "freq_meanstd"],
                    help="mean = moyenne de tous les tokens AST ; freq = garde les "
                         "bandes de frequence separees et score bande par bande")
    ap.add_argument("--pseudo-labels", type=int, default=0,
                    help="nombre de pseudo-classes creees par k-means pour les machines "
                         "sans attributs (0 = desactive). Entierement non supervise.")
    ap.add_argument("--domain-align", action="store_true",
                    help="recentre les references cible sur la moyenne source")
    ap.add_argument("--threshold", default="gamma",
                    choices=["gamma", "percentile", "mixture", "otsu"],
                    help="strategie de calibration du seuil normal/anomalie")
    ap.add_argument("--threshold-q", type=float, default=0.9,
                    help="quantile pour les strategies gamma et percentile")
    ap.add_argument("--farmix", type=int, default=0,
                    help="nombre de copies bruitees ajoutees a la banque de reference, "
                         "le bruit venant du canal 2 d'autres clips (0 = desactive)")
    ap.add_argument("--farmix-snr-min", type=float, default=0.0)
    ap.add_argument("--farmix-snr-max", type=float, default=20.0)
    ap.add_argument("--per-domain", action="store_true",
                    help="cherche le voisin separement par domaine et prend le minimum ; "
                         "traite le desequilibre 990 source / 10 cible")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-deterministic", dest="deterministic", action="store_false",
                    help="laisse cuDNN choisir des algorithmes non deterministes")
    ap.set_defaults(deterministic=True)
    ap.add_argument("--cache-dir", default="feat_cache")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    # Les convolutions cuDNN sont non deterministes par defaut : a graine fixee,
    # deux executions du CNN peuvent differer de plusieurs points. On force le
    # mode deterministe pour que les chiffres rapportes soient reproductibles.
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as e:
            print(f"  mode deterministe partiel : {e}")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)

    machines = discover(args.data_root)
    if args.machines:
        keep = {m.strip() for m in args.machines.split(",")}
        machines = {k: v for k, v in machines.items() if k in keep}
    if not machines:
        sys.exit(f"Aucune machine trouvee sous {args.data_root}")
    print(f"{len(machines)} machines : {', '.join(machines)}")
    print(f"frontend={args.frontend} masque={args.mask} alpha={args.alpha} "
          f"beta={args.beta} seed={args.seed} device={device}\n")

    fb = mel_filterbank(16000, N_FFT, N_MELS, FMIN)

    t0 = time.time()
    print("Features...")
    data = {}
    for m, splits in machines.items():
        d = build_features(m, splits, args, fb, cache_dir)
        if "train_X" not in d or "test_X" not in d:
            print(f"  {m}: train ou test manquant, ignore")
            continue
        d["_scorable"] = "anomaly" in set(d["test_lab"])
        data[m] = d
        attrs = sorted(set(d["train_attr"]))
        att = f"{len(attrs)} attribut(s)" + ("" if len(attrs) > 1 else f" [{attrs[0]}]")
        flag = "" if d["_scorable"] else "  [non labellise -> export seul]"
        print(f"  {m}: train {d['train_X'].shape}  test {d['test_X'].shape}  {att}{flag}")
    if not data:
        sys.exit("Aucune machine exploitable.")
    print(f"  ({time.time()-t0:.0f} s)\n")

    # Quatre machines du dev n'ont qu'un seul attribut : le reseau n'aurait
    # qu'a reconnaitre de quelle machine il s'agit, ce qui est trivial et
    # produit des embeddings peu structures. On recree des sous-groupes par
    # k-means sur le profil spectral moyen. Aucune etiquette n'est utilisee.
    if args.pseudo_labels > 0 and args.frontend == "cnn":
        print("Pseudo-etiquettes :")
        for m, d in data.items():
            if len(set(d["train_attr"])) > 1:
                print(f"  {m}: {len(set(d['train_attr']))} attributs reels, inchange")
                continue
            V = d["train_X"].mean(axis=2)
            V = (V - V.mean(0)) / (V.std(0) + 1e-6)
            k = min(args.pseudo_labels, len(V))
            lab = KMeans(n_clusters=k, n_init=10, random_state=args.seed).fit_predict(V)
            d["train_attr"] = np.array([f"pseudo_{i:02d}" for i in lab])
            sizes = np.bincount(lab)
            print(f"  {m}: {k} pseudo-classes, tailles {sizes.min()}-{sizes.max()}")
        print()

    if args.frontend == "ast":
        print("Front-end gele : AST pre-entraine, aucun entrainement.")
        t0 = time.time()
        extractor = FrozenAST(device, args.pooling)
        train_time = 0.0
        n_params = extractor.n_params
        embed_fn = extractor.embed
        print(f"  checkpoint charge en {time.time()-t0:.0f} s, "
              f"{n_params/1e6:.1f} M parametres geles\n")
    else:
        # Classes = machine x attributs. Un modele unique pour toutes les machines.
        classes = sorted({f"{m}|{a}" for m, d in data.items() for a in d["train_attr"]})
        cls_idx = {c: i for i, c in enumerate(classes)}
        Xtr = [x for m, d in data.items() for x in d["train_X"]]
        ytr = np.array([cls_idx[f"{m}|{a}"] for m, d in data.items() for a in d["train_attr"]])
        print(f"Entrainement : {len(Xtr)} clips, {len(classes)} classes")

        t0 = time.time()
        model = train_model(Xtr, ytr, len(classes), args, device)
        train_time = time.time() - t0
        n_params = sum(p.numel() for p in model.parameters())
        embed_fn = lambda X: embed_all(model, X, device)
        print(f"  {train_time:.0f} s, {n_params/1e6:.2f} M parametres\n")

    print("Evaluation...")
    per_machine, rows, raw_scores = {}, [], {}
    t0 = time.time()
    for m, d in data.items():
        Etr = embed_fn(d["train_X"])
        dom_tr = d["train_dom"]
        if args.farmix > 0 and "train_N" in d:
            banks, doms = [Etr], [dom_tr]
            for _ in range(args.farmix):
                Xa = farmix_round(d["train_X"], d["train_N"], args.farmix_snr_min,
                                  args.farmix_snr_max, rng, args.frontend)
                banks.append(embed_fn(Xa))
                doms.append(dom_tr)
                del Xa
            Etr, dom_tr = np.concatenate(banks), np.concatenate(doms)
        Ete = embed_fn(d["test_X"])
        s = knn_scores(Etr, dom_tr, Ete, args.knn, args.domain_align,
                       args.band_agg, args.per_domain)
        tr = knn_scores(Etr, dom_tr, Etr, args.knn, args.domain_align,
                        args.band_agg, args.per_domain, skip_self=True)
        thr = choose_threshold(tr, s, args.threshold, args.threshold_q)
        if args.export_scores:
            export_dcase_csv(m, d, s, thr, args, outdir)
        if not d["_scorable"]:
            print("  " + m.ljust(16) + "scores exportes (pas de label disponible)")
            continue
        mm = machine_metrics(d["test_dom"], d["test_lab"], s, thr)
        per_machine[m] = mm
        raw_scores[m] = dict(score=s, dom=d["test_dom"], lab=d["test_lab"])
        rows.append(dict(machine=m, **{k: round(v, 2) for k, v in mm.items()}))
        print("  " + m.ljust(16) + "  ".join(f"{k} {v:6.2f}" for k, v in mm.items()))
    infer_time = (time.time() - t0) / sum(len(d["test_X"]) for d in data.values())

    score = official_score(per_machine)
    print("\n" + "=" * 62)
    if per_machine:
        print(f"SCORE OFFICIEL : {score:.2f}")
        print("  references dev : AE 56.66 | Mahalanobis 57.66 | BEATs gele 60.28 | NA-BEATs 63.18")
    else:
        print("Aucune machine labellisee : pas de metrique calculable ici.")
        print("  Les scores sont exportes, passez-les a l'evaluateur officiel")
        print("  (depot nttcslab/dcase2026_task2_evaluator) pour obtenir AUC, pAUC et F1.")
    print(f"  parametres {n_params/1e6:.2f} M | entrainement {train_time:.0f} s | "
          f"inference {infer_time*1000:.1f} ms/clip")
    if args.export_scores:
        print(f"  CSV de soumission : {outdir}/dcase_submission/")

    # Le tag doit encoder TOUT parametre qui change le resultat, sinon deux
    # configurations differentes ecrivent dans le meme fichier et l'une ecrase
    # l'autre en silence. Le jeu de donnees en fait partie : un run dev et un
    # run eval de meme configuration produisent des resultats distincts.
    ds = Path(args.data_root).name or "data"
    parts = [ds, args.frontend, args.mask,
             f"a{args.alpha:g}", f"b{args.beta:g}", f"c{args.calib_pct:g}",
             f"gf{args.gfloor:g}", f"knn{args.knn}", f"th-{args.threshold}{args.threshold_q:g}"]
    if args.frontend == "ast":
        parts.append(f"pool-{args.pooling}")
    parts.append(f"agg-{args.band_agg}")
    if args.per_domain:
        parts.append("perdom")
    if args.domain_align:
        parts.append("align")
    if args.swap_channels:
        parts.append("swap")
    if args.farmix:
        parts.append(f"fm{args.farmix}snr{args.farmix_snr_min:g}-{args.farmix_snr_max:g}")
    if args.frontend == "cnn":
        parts += [f"ep{args.epochs}", f"crop{args.crop}", f"sa{args.specaug:g}",
                  f"ls{args.label_smoothing:g}", f"pl{args.pseudo_labels}"]
    parts.append(f"s{args.seed}")
    tag = "_".join(parts)

    np.savez_compressed(outdir / f"scores_{tag}.npz",
                        **{f"{m}__{k}": v for m, d in raw_scores.items() for k, v in d.items()})
    with open(outdir / f"metrics_{tag}.csv", "w", newline="") as fh:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(fh, fieldnames=["machine"] + [k for k in keys if k != "machine"])
        w.writeheader(); w.writerows(rows)
    with open(outdir / f"summary_{tag}.json", "w") as fh:
        cfg = dict(vars(args))
        cfg["n_params"] = n_params
        cfg["dataset"] = Path(args.data_root).name
        json.dump(dict(config=cfg, official_score=score, per_machine=per_machine,
                       n_params=n_params, train_time_s=train_time,
                       infer_ms_per_clip=infer_time * 1000), fh, indent=2, default=str)
    print(f"\nResultats dans {outdir}/ (tag {tag})")


if __name__ == "__main__":
    main()