#!/usr/bin/env python3
"""
Agrege tous les resultats en deux fichiers exploitables.

  results_dev.csv    une ligne par configuration testee sur le developpement
  results_eval.csv   une ligne par soumission evaluee par l'evaluateur officiel

Les metriques du developpement sont RECALCULEES a partir des scores bruts
(.npz) plutot que relues dans les JSON : certains JSON produits avant
correction contiennent un score officiel contamine par les colonnes
precision/rappel/F1.

Usage :
    python collect_results.py
    python collect_results.py --results asd_results --evaluator dcase2026_task2_evaluator
"""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

KEEP = ("AUC_source", "AUC_target", "pAUC")


def pauc(y, s, p=0.1):
    y, s = np.asarray(y), np.asarray(s)
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, s, max_fpr=p) * 100


def metrics_from_scores(dom, lab, score):
    y = (lab == "anomaly").astype(int)
    out = {}
    for d in ("source", "target"):
        m = dom == d
        if m.sum() and len(np.unique(y[m])) == 2:
            out[f"AUC_{d}"] = roc_auc_score(y[m], score[m]) * 100
    out["pAUC"] = pauc(y, score)
    return out


def official(per_machine):
    vals = [v for d in per_machine.values() for k, v in d.items()
            if k in KEEP and v == v]
    return len(vals) / np.sum(1.0 / np.maximum(vals, 1e-6)) if vals else float("nan")


def read_dev(results_dir):
    """Recalcule les metriques de chaque run a partir de ses scores bruts."""
    rows = []
    for npz_path in sorted(Path(results_dir).glob("scores_*.npz")):
        if npz_path.stat().st_size < 200:
            print(f"  ignore {npz_path.name} : fichier vide (jeu sans label)")
            continue
        tag = npz_path.stem[len("scores_"):]
        cfg_path = Path(results_dir) / f"summary_{tag}.json"
        cfg, js_per_machine = {}, {}
        if cfg_path.exists():
            j = json.loads(cfg_path.read_text())
            cfg = dict(j.get("config", {}))
            # n_params est ecrit a la racine du JSON par les versions anterieures
            cfg.setdefault("n_params", j.get("n_params", ""))
            js_per_machine = j.get("per_machine", {})

        z = np.load(npz_path, allow_pickle=True)
        machines = sorted({k.split("__")[0] for k in z.files})
        per_machine, flat = {}, {}
        for m in machines:
            try:
                mm = metrics_from_scores(z[f"{m}__dom"], z[f"{m}__lab"], z[f"{m}__score"])
            except KeyError:
                continue
            per_machine[m] = mm
            for k, v in mm.items():
                flat[f"{m}_{k}"] = round(v, 2)
            # precision / rappel / F1 dependent du seuil, donc absents du
            # recalcul : on les reprend du JSON du run.
            for k in ("prec", "rec", "F1"):
                v = js_per_machine.get(m, {}).get(k)
                if v is not None:
                    flat[f"{m}_{k}"] = round(float(v), 2)
        if not per_machine:
            continue

        rows.append(dict(
            tag=tag,
            frontend=cfg.get("frontend", ""), mask=cfg.get("mask", ""),
            pooling=cfg.get("pooling", ""), band_agg=cfg.get("band_agg", ""),
            per_domain=cfg.get("per_domain", ""), domain_align=cfg.get("domain_align", ""),
            knn=cfg.get("knn", ""), alpha=cfg.get("alpha", ""), beta=cfg.get("beta", ""),
            swap_channels=cfg.get("swap_channels", ""),
            farmix=cfg.get("farmix", ""),
            farmix_snr=f'{cfg.get("farmix_snr_min","")}-{cfg.get("farmix_snr_max","")}',
            threshold=cfg.get("threshold", ""), threshold_q=cfg.get("threshold_q", ""),
            pseudo_labels=cfg.get("pseudo_labels", ""), seed=cfg.get("seed", ""),
            dataset=cfg.get("dataset", ""),
            n_params=cfg.get("n_params", ""),
            official_score=round(official(per_machine), 2),
            **flat))
    return rows


def read_eval(evaluator_dir):
    """Lit les CSV produits par l'evaluateur officiel DCASE."""
    rows = []
    res_dir = Path(evaluator_dir) / "teams_result"
    for f in sorted(res_dir.glob("*_result.csv")):
        name = f.stem.replace("_result", "")
        txt = f.read_text()
        row = {"submission": name}

        m = re.search(r"official score,,([\d.eE+-]+)", txt)
        if m:
            row["official_score"] = round(float(m.group(1)) * 100, 2)

        for lbl, prefix in [("harmonic mean over all", "hm"),
                            ("source harmonic mean over all", "src"),
                            ("target harmonic mean over all", "tgt")]:
            m = re.search(r'"' + lbl + r'[^"]*",,' + r",".join([r"([\d.eE+-]+)"] * 5), txt)
            if m:
                for k, v in zip(("AUC", "pAUC", "prec", "rec", "F1"), m.groups()):
                    row[f"{prefix}_{k}"] = round(float(v) * 100, 2)

        # bloc par machine : nom de machine puis ligne "00,..."
        for mm in re.finditer(r"^([A-Za-z][\w]*)\nsection,[^\n]*\n00,([^\n]*)$",
                              txt, re.MULTILINE):
            machine, vals = mm.group(1), mm.group(2).split(",")
            names = ["AUC_all", "AUC_source", "AUC_target", "pAUC",
                     "prec_source", "prec_target", "rec_source", "rec_target",
                     "F1_source", "F1_target"]
            for k, v in zip(names, vals):
                try:
                    row[f"{machine}_{k}"] = round(float(v) * 100, 2)
                except ValueError:
                    pass
        rows.append(row)
    return rows


def write_csv(path, rows):
    if not rows:
        print(f"  (rien a ecrire dans {path})")
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  {path} : {len(rows)} lignes, {len(keys)} colonnes")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="asd_results")
    ap.add_argument("--evaluator", default="dcase2026_task2_evaluator")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    out = Path(args.out)
    print("Agregation :")
    dev = read_dev(args.results)
    write_csv(out / "results_dev.csv", dev)
    ev = read_eval(args.evaluator)
    write_csv(out / "results_eval.csv", ev)

    if dev:
        best = max(dev, key=lambda r: r["official_score"])
        print(f"\nMeilleure config dev : {best['official_score']}  [{best['tag']}]")
    if ev:
        scored = [r for r in ev if "official_score" in r]
        if scored:
            b = max(scored, key=lambda r: r["official_score"])
            print(f"Meilleure soumission eval : {b['official_score']}  [{b['submission']}]")


if __name__ == "__main__":
    main()