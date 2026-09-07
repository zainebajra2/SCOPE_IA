#!/usr/bin/env python3
"""
E0b - Validation physique de la coherence inter-canaux, SANS le dossier
      /supplemental (absent des archives livrees).
DCASE 2026 Task 2 (Noise-aware UASD), development dataset (Zenodo 19336329).

Pourquoi cette seconde version
------------------------------
E0 mesurait Q2 par contrast_weighted() : MSC ponderee par l'energie, comparee
a la MSC des bins de faible energie. Cette metrique confond deux effets. Les
bins de faible energie sont souvent en haute frequence, ou la MSC chute pour
des raisons purement geometriques (dephasage entre micros), independamment de
ce qui emet. Le "bruit" mesure n'est donc pas du bruit : c'est de la haute
frequence. D'ou le contraste negatif absurde sur ToyCarEmu (-0.275).

E0b remplace cette mesure par deux estimateurs, chacun controlant le
confondant frequentiel :

  A. CONTRASTE TEMPOREL (machines intermittentes)
     Dans un meme clip, comparer la MSC des trames ou la machine emet a celle
     des trames ou elle se tait. Memes micros, meme salle, memes frequences.
     Seule la presence de la machine change. C'est le substitut le plus proche
     de /supplemental. Applicable si la profondeur de modulation temporelle
     depasse MOD_MIN_DB.

  B. CONTRASTE SPECTRAL LOCAL (toutes machines, y compris stationnaires)
     Comparer la MSC des raies tonales (machine) a celle du plancher spectral
     voisin (bruit), bande par bande. La comparaison etant faite a l'interieur
     de chaque bande, la dependance en frequence de la MSC ne biaise plus le
     resultat. Estimateur plus faible que A, mais applicable a fan, bearing et
     gearbox qui ne s'interrompent jamais.

Q1 est restreint aux bins ou la machine emet. En large bande, le micro
lointain peut etre plus energetique simplement parce qu'il est plus pres des
haut-parleurs de bruit : E0 signalait a tort une inversion de canaux sur fan
(-2.8 dB) et sliderEmu (-0.8 dB).

Q5 est retrograde en diagnostic, hors du verdict. Le bruit d'usine a ete
rejoue par quatre haut-parleurs aux coins de la salle. Quatre sources
ponctuelles ne forment pas un champ diffus, donc l'ajustement sinc^2 n'a
aucune raison de converger et son echec ne dit rien sur la validite du masque.

Dependances : numpy, scipy, matplotlib.

Usage :
    python3 e0b_coherence_revised.py \
        --data-root /mnt/storage_1_10T/scope/data/DCASE2026_Task2/dev \
        --out results_e0b
"""

import argparse
import csv
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.ndimage import median_filter

C_SOUND = 343.0
SPLITS = ("train", "test", "supplemental")
EPS = 1e-20

# Seuils
MOD_MIN_DB = 6.0    # modulation temporelle mini pour appliquer la methode A
PEAK_DB = 3.0       # proeminence mini d'une raie au-dessus du plancher local
F_MIN = 200.0       # sous ce seuil le bruit diffus est deja coherent
CONTRAST_OK = 0.15  # seuil de decision sur le contraste

BANDS = [(200, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]


# --------------------------------------------------------------------------
# Chargement
# --------------------------------------------------------------------------

def read_stereo(path):
    """(x1, x2, fs) en float64. x1 = canal 1 (cense etre le micro proche)."""
    fs, data = wavfile.read(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"{data.ndim}D shape={data.shape}, 2 canaux attendus")
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float64) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float64)
    return data[:, 0], data[:, 1], fs


def discover(data_root):
    """{machine_type: {split: [paths]}} a n'importe quelle profondeur."""
    found = {}
    for dirpath, dirnames, _ in os.walk(data_root):
        d = Path(dirpath)
        if d.name in SPLITS:
            continue
        splits = {}
        for split in SPLITS:
            if split in dirnames:
                wavs = sorted((d / split).glob("*.wav"))
                if wavs:
                    splits[split] = wavs
        if splits:
            found[d.name] = splits
            dirnames[:] = [x for x in dirnames if x not in SPLITS]
    return dict(sorted(found.items()))


def tag_of(path):
    n = path.name.lower()
    dom = "source" if "source" in n else ("target" if "target" in n else "unknown")
    lab = "anomaly" if "anomaly" in n else ("normal" if "normal" in n else "unknown")
    return dom, lab


# --------------------------------------------------------------------------
# STFT et coherence sur sous-ensemble de trames
# --------------------------------------------------------------------------

def stft(x, nperseg, hop, window):
    """(n_frames, n_bins) complexe."""
    n_frames = 1 + (len(x) - nperseg) // hop
    if n_frames < 1:
        return None
    idx = np.arange(nperseg)[None, :] + hop * np.arange(n_frames)[:, None]
    return np.fft.rfft(x[idx] * window[None, :], axis=1)


def msc_over(X1, X2, sel):
    """
    MSC estimee sur les trames selectionnees.

    C'est la definition de Welch appliquee a un sous-ensemble : moyenne du
    spectre croise divisee par le produit des auto-spectres. Selectionner les
    trames avant de moyenner est exactement ce qui permet d'isoler la machine
    sans disposer d'enregistrements separes.
    """
    if sel.sum() < 8:
        return None
    a, b = X1[sel], X2[sel]
    s12 = np.mean(a * np.conj(b), axis=0)
    s11 = np.mean(np.abs(a) ** 2, axis=0)
    s22 = np.mean(np.abs(b) ** 2, axis=0)
    return np.abs(s12) ** 2 / (s11 * s22 + EPS)


def clip_measures(path, nperseg, hop, window, frac):
    x1, x2, fs = read_stereo(path)
    n = min(len(x1), len(x2))
    if n < 8 * nperseg:
        return None
    x1, x2 = x1[:n], x2[:n]

    X1, X2 = stft(x1, nperseg, hop, window), stft(x2, nperseg, hop, window)
    if X1 is None or X2 is None:
        return None
    f = np.fft.rfftfreq(nperseg, 1.0 / fs)

    # Energie par trame sur le canal proche, au-dessus de F_MIN pour ne pas
    # laisser le rumble basse frequence piloter la detection d'activite.
    hf = f >= F_MIN
    e_frame = np.mean(np.abs(X1[:, hf]) ** 2, axis=1)
    k = max(8, int(len(e_frame) * frac))
    order = np.argsort(e_frame)
    sel_sil = np.zeros(len(e_frame), bool); sel_sil[order[:k]] = True
    sel_act = np.zeros(len(e_frame), bool); sel_act[order[-k:]] = True

    mod_db = 10 * np.log10(np.mean(e_frame[sel_act]) / (np.mean(e_frame[sel_sil]) + EPS))

    return dict(
        f=f, fs=fs, n_frames=X1.shape[0], mod_db=float(mod_db),
        msc_all=msc_over(X1, X2, np.ones(X1.shape[0], bool)),
        msc_act=msc_over(X1, X2, sel_act),
        msc_sil=msc_over(X1, X2, sel_sil),
        p1=np.mean(np.abs(X1) ** 2, axis=0),
        p2=np.mean(np.abs(X2) ** 2, axis=0),
        p1_act=np.mean(np.abs(X1[sel_act]) ** 2, axis=0),
        p2_act=np.mean(np.abs(X2[sel_act]) ** 2, axis=0),
    )


def collect(paths, nperseg, hop, window, frac, n_clips, rng):
    if not paths:
        return None
    if len(paths) > n_clips:
        paths = [paths[i] for i in rng.choice(len(paths), n_clips, replace=False)]
    acc = {k: [] for k in ("msc_all", "msc_act", "msc_sil",
                           "p1", "p2", "p1_act", "p2_act")}
    mods, f, fs, nf = [], None, None, []
    for p in paths:
        try:
            m = clip_measures(p, nperseg, hop, window, frac)
        except Exception as e:
            warnings.warn(f"{p.name}: {e}")
            continue
        if m is None:
            continue
        f, fs = m["f"], m["fs"]
        mods.append(m["mod_db"]); nf.append(m["n_frames"])
        for k in acc:
            if m[k] is not None:
                acc[k].append(m[k])
    if not acc["msc_all"]:
        return None
    out = dict(f=f, fs=fs, n=len(acc["msc_all"]),
               mod_db=float(np.median(mods)), n_frames=int(np.median(nf)))
    for k, v in acc.items():
        out[k] = np.median(np.vstack(v), axis=0) if v else None
    return out


# --------------------------------------------------------------------------
# Detection des raies machine
# --------------------------------------------------------------------------

def peak_mask(f, psd, peak_db=PEAK_DB, width=21):
    """
    Bins ou la PSD depasse nettement son plancher local.

    Le bruit d'usine est large bande ; les composantes machine (harmoniques de
    rotation, resonances) sont des raies. Un filtre median en frequence estime
    le plancher, ce qui depasse le plancher de peak_db est attribue a la
    machine. Approximation, mais elle ne suppose rien sur les canaux, donc
    elle ne biaise pas Q1.
    """
    logp = 10 * np.log10(psd + EPS)
    env = median_filter(logp, size=width, mode="nearest")
    return (logp - env > peak_db) & (f >= F_MIN)


# --------------------------------------------------------------------------
# Q1, Q2
# --------------------------------------------------------------------------

def q1_band(f, p1, p2, peaks):
    """Rapport d'energie ch1/ch2 restreint aux bins machine."""
    wide = 10 * np.log10(np.sum(p1[f >= F_MIN]) / (np.sum(p2[f >= F_MIN]) + EPS))
    if peaks.sum() < 5:
        return wide, None
    band = 10 * np.log10(np.sum(p1[peaks]) / (np.sum(p2[peaks]) + EPS))
    return wide, band


def q2_temporal(msc_act, msc_sil, f):
    """Methode A : trames actives contre trames silencieuses."""
    if msc_act is None or msc_sil is None:
        return None
    sel = f >= F_MIN
    hi, lo = float(np.mean(msc_act[sel])), float(np.mean(msc_sil[sel]))
    return hi, lo, hi - lo


def q2_spectral_local(f, msc, peaks):
    """
    Methode B : raies contre plancher, bande par bande.

    La comparaison reste a l'interieur d'une meme bande, donc la decroissance
    naturelle de la MSC avec la frequence s'annule entre les deux termes. C'est
    la correction du defaut de contrast_weighted().
    """
    floor = (~peaks) & (f >= F_MIN)
    hs, ls, ws = [], [], []
    for lo_f, hi_f in BANDS:
        b = (f >= lo_f) & (f < hi_f)
        pk, fl = b & peaks, b & floor
        if pk.sum() >= 3 and fl.sum() >= 10:
            hs.append(np.mean(msc[pk])); ls.append(np.mean(msc[fl]))
            ws.append(pk.sum())
    if not ws:
        return None
    w = np.array(ws, float); w /= w.sum()
    hi = float(np.sum(w * np.array(hs)))
    lo = float(np.sum(w * np.array(ls)))
    return hi, lo, hi - lo


# --------------------------------------------------------------------------
# Q5 : diagnostic seulement
# --------------------------------------------------------------------------

def diffuse_msc(f, d):
    x = 2.0 * np.pi * f * d / C_SOUND
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.where(x == 0, 1.0, np.sin(x) / x)
    return s ** 2


def estimate_spacing(f, curve, f_max=4000.0):
    sel = (f > 0) & (f <= f_max)
    fv, yv = f[sel], curve[sel]
    grid = np.arange(0.02, 1.01, 0.002)
    err = np.array([np.mean((yv - diffuse_msc(fv, d)) ** 2) for d in grid])
    k = int(np.argmin(err))
    d_hat, mse = float(grid[k]), float(err[k])
    ok = (mse < 0.02) and (grid[0] + 1e-9 < d_hat < grid[-1] - 1e-9)
    return d_hat, mse, ok


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def fig_machine(machine, ref, peaks, outdir):
    f = ref["f"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    ax = axes[0]
    if ref["msc_act"] is not None:
        ax.plot(f, ref["msc_act"], lw=1.2, color="tab:red", label="trames actives")
    if ref["msc_sil"] is not None:
        ax.plot(f, ref["msc_sil"], lw=1.2, color="tab:blue", label="trames silencieuses")
    ax.plot(f, ref["msc_all"], lw=0.9, color="0.4", ls="--", label="toutes trames")
    ax.axvline(F_MIN, color="0.6", lw=0.8, ls=":")
    ax.set_ylim(0, 1.02); ax.set_ylabel(r"MSC $\gamma^2$")
    ax.set_title(f"{machine} - contraste temporel (modulation {ref['mod_db']:.1f} dB)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(f, ref["msc_all"], lw=0.9, color="0.4")
    if peaks.any():
        ax.plot(f[peaks], ref["msc_all"][peaks], ".", ms=4, color="tab:red",
                label=f"raies machine (n={peaks.sum()})")
    ax.axvline(F_MIN, color="0.6", lw=0.8, ls=":")
    ax.set_ylim(0, 1.02); ax.set_ylabel(r"MSC $\gamma^2$")
    ax.set_title("contraste spectral local : raies contre plancher")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(f, 10 * np.log10(ref["p1"] + EPS), lw=1.0, label="canal 1")
    ax.plot(f, 10 * np.log10(ref["p2"] + EPS), lw=1.0, label="canal 2")
    if peaks.any():
        ax.plot(f[peaks], 10 * np.log10(ref["p1"][peaks] + EPS), ".", ms=4,
                color="tab:red")
    ax.set_xlabel("Frequence (Hz)"); ax.set_ylabel("PSD (dB)")
    ax.set_xscale("log"); ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(outdir / f"e0b_{machine}.png", dpi=130)
    plt.close(fig)


def fig_summary(rows, outdir):
    fig, ax = plt.subplots(figsize=(9, 5))
    names = [r["machine"] for r in rows]
    y = np.arange(len(names))
    tmp = [r["contraste_temporel"] if r["contraste_temporel"] != "" else np.nan for r in rows]
    spc = [r["contraste_spectral"] if r["contraste_spectral"] != "" else np.nan for r in rows]
    ax.barh(y - 0.2, tmp, 0.4, label="temporel (methode A)", color="tab:red")
    ax.barh(y + 0.2, spc, 0.4, label="spectral local (methode B)", color="tab:blue")
    ax.axvline(CONTRAST_OK, color="k", ls="--", lw=1, label=f"seuil {CONTRAST_OK}")
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_xlabel("contraste MSC machine - bruit")
    ax.set_title("E0b - contraste par machine et par methode")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(outdir / "e0b_summary.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="results_e0b")
    ap.add_argument("--n-clips", type=int, default=60)
    ap.add_argument("--nperseg", type=int, default=1024)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--frac", type=float, default=0.25,
                    help="fraction des trames prises comme actives / silencieuses")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    hop = max(1, int(args.nperseg * (1 - args.overlap)))
    window = np.hanning(args.nperseg)
    rng = np.random.default_rng(args.seed)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    machines = discover(args.data_root)
    if not machines:
        sys.exit(f"Aucun dossier machine sous {args.data_root}")
    print(f"{len(machines)} types de machine : {', '.join(machines)}")
    print(f"nperseg={args.nperseg}  hop={hop}  frac={args.frac}\n")

    rows = []

    for machine, splits in machines.items():
        train = splits.get("train", [])
        test = splits.get("test", [])
        src = [p for p in train if tag_of(p)[0] == "source"] or train
        tgt = [p for p in train if tag_of(p)[0] == "target"]
        t_nrm = [p for p in test if tag_of(p)[1] == "normal"]
        t_ano = [p for p in test if tag_of(p)[1] == "anomaly"]

        ref = collect(src, args.nperseg, hop, window, args.frac, args.n_clips, rng)
        if ref is None:
            print(f"[{machine}] aucun clip exploitable, ignore\n")
            continue
        f = ref["f"]
        peaks = peak_mask(f, ref["p1_act"] if ref["p1_act"] is not None else ref["p1"])

        n_sel = max(8, int(ref["n_frames"] * args.frac))
        print(f"[{machine}]  fs={ref['fs']} Hz  trames/clip={ref['n_frames']}  "
              f"biais MSC~{1/n_sel:.3f}  raies detectees={int(peaks.sum())}")

        # Q1
        wide, band = q1_band(f, ref["p1"], ref["p2"], peaks)
        if band is None:
            print(f"   Q1 energie ch1/ch2 : {wide:+.1f} dB large bande "
                  f"(pas assez de raies pour restreindre)")
            q1_val = wide
        else:
            verdict = "ch1 = proche, OK" if band > 0 else "!! ch2 plus fort sur les raies machine"
            print(f"   Q1 energie ch1/ch2 : {band:+.1f} dB sur les raies "
                  f"({wide:+.1f} dB en large bande) -> {verdict}")
            q1_val = band

        # Q2 methode A
        ta = q2_temporal(ref["msc_act"], ref["msc_sil"], f)
        impulsive = ref["mod_db"] >= MOD_MIN_DB
        if ta is not None and impulsive:
            hi, lo, ctr_t = ta
            print(f"   Q2-A temporel [modulation {ref['mod_db']:.1f} dB] : "
                  f"actif {hi:.3f} / silence {lo:.3f} -> ecart {ctr_t:+.3f} "
                  + ("OK" if ctr_t > CONTRAST_OK else "FAIBLE"))
        else:
            ctr_t = None
            print(f"   Q2-A temporel NON APPLICABLE (modulation {ref['mod_db']:.1f} dB "
                  f"< {MOD_MIN_DB} dB : machine stationnaire, pas de trame sans machine)")

        # Q2 methode B
        sp = q2_spectral_local(f, ref["msc_all"], peaks)
        if sp is not None:
            hi_s, lo_s, ctr_s = sp
            print(f"   Q2-B spectral local : raies {hi_s:.3f} / plancher {lo_s:.3f} "
                  f"-> ecart {ctr_s:+.3f} "
                  + ("OK" if ctr_s > CONTRAST_OK else "FAIBLE"))
        else:
            ctr_s = None
            print("   Q2-B spectral local NON APPLICABLE (pas assez de raies par bande)")

        # Q4
        r_t = collect(tgt, args.nperseg, hop, window, args.frac, args.n_clips, rng)
        drift = ""
        if r_t is not None:
            drift = float(np.mean(np.abs(r_t["msc_all"] - ref["msc_all"])))
            print(f"   Q4 derive source->cible : {drift:.3f} "
                  + ("OK" if drift < 0.10 else "ATTENTION"))
            drift = round(drift, 4)

        # Q5 diagnostic
        base = ref["msc_sil"] if ref["msc_sil"] is not None else ref["msc_all"]
        d_hat, d_err, d_ok = estimate_spacing(f, base)
        if d_ok:
            print(f"   Q5 [diagnostic] espacement compatible d = {d_hat*100:.1f} cm "
                  f"(mse={d_err:.4f})")
        else:
            print(f"   Q5 [diagnostic] pas de champ diffus (mse={d_err:.4f}) - attendu, "
                  f"le bruit vient de 4 haut-parleurs ponctuels")

        # bonus anomalie
        r_n = collect(t_nrm, args.nperseg, hop, window, args.frac, args.n_clips, rng)
        r_a = collect(t_ano, args.nperseg, hop, window, args.frac, args.n_clips, rng)
        if r_n is not None and r_a is not None:
            print(f"   bonus  MSC anomalie - normal : "
                  f"{np.mean(r_a['msc_all'] - r_n['msc_all']):+.3f}")
        print()

        rows.append(dict(
            machine=machine, n_clips=ref["n"], fs=ref["fs"],
            trames_par_clip=ref["n_frames"], raies=int(peaks.sum()),
            modulation_dB=round(ref["mod_db"], 2),
            regime="impulsif" if impulsive else "stationnaire",
            q1_raies_dB=round(q1_val, 2),
            contraste_temporel=round(ctr_t, 4) if ctr_t is not None else "",
            contraste_spectral=round(ctr_s, 4) if ctr_s is not None else "",
            derive_source_cible=drift,
            d_diagnostic_cm=round(d_hat * 100, 1) if d_ok else "",
        ))
        fig_machine(machine, ref, peaks, outdir)

    if not rows:
        sys.exit("Rien a analyser.")

    fig_summary(rows, outdir)
    csv_path = outdir / "e0b_summary.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ----------------------------------------------------------------------
    # Verdict
    # ----------------------------------------------------------------------
    imp = [r for r in rows if r["regime"] == "impulsif"]
    sta = [r for r in rows if r["regime"] == "stationnaire"]
    ct = [r["contraste_temporel"] for r in imp if r["contraste_temporel"] != ""]
    cs = [r["contraste_spectral"] for r in rows if r["contraste_spectral"] != ""]

    print("=" * 72)
    print("VERDICT E0b")
    print(f"  {len(imp)} machines impulsives, {len(sta)} stationnaires")
    if ct:
        ok = sum(1 for c in ct if c > CONTRAST_OK)
        print(f"  Methode A (preuve forte)  : median {np.median(ct):+.3f}, "
              f"{ok}/{len(ct)} au-dessus de {CONTRAST_OK}")
    else:
        print("  Methode A : aucune machine assez modulee")
    if cs:
        ok = sum(1 for c in cs if c > CONTRAST_OK)
        print(f"  Methode B (preuve faible) : median {np.median(cs):+.3f}, "
              f"{ok}/{len(cs)} au-dessus de {CONTRAST_OK}")

    strong = bool(ct) and np.median(ct) > CONTRAST_OK
    broad = bool(cs) and sum(1 for c in cs if c > CONTRAST_OK) >= len(cs) / 2
    print()
    if strong and broad:
        print("  -> GO. Le contraste tient sur les deux estimateurs.")
        print("     Passer a E1 : reproduire la baseline mono-canal.")
    elif strong:
        print("  -> GO CIBLE. La preuve forte ne porte que sur les machines")
        print("     impulsives. C'est defendable, mais la these du papier doit")
        print("     etre restreinte a ce regime plutot que presentee comme generale.")
    elif broad:
        print("  -> GO PRUDENT. Seule la methode B est positive, or elle repose sur")
        print("     l'hypothese que les raies viennent de la machine. A confirmer")
        print("     avant d'engager E2 : verifier sur les figures que les raies")
        print("     detectees correspondent aux harmoniques attendues.")
    else:
        print("  -> NO GO en l'etat. Avant d'abandonner : nperseg 2048 ou 4096 pour")
        print("     mieux resoudre les raies, ou --frac 0.15 pour des trames actives")
        print("     plus selectives. Si le contraste ne monte pas, le masque MSC")
        print("     n'a pas de fondement sur ce dataset.")
    print(f"\nFigures et {csv_path.name} dans {outdir}/")


if __name__ == "__main__":
    main()
