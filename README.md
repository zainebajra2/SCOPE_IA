# SCOPE_IA

## Scripts (ordre d'exécution)

1. **`e0_coherence_analysis.py`** *(optionnel)* — diagnostic préalable : vérifie la cohérence physique entre les deux canaux audio du dataset dev.
2. **`run_all.sh`** — lance toutes les configurations de la pipeline (dev puis eval) en appelant `asd_pipeline.py` en boucle.
   - **`asd_pipeline.py`** — cœur de la chaîne ASD (frontend CNN ou AST + masque inter-canaux optionnel) ; peut aussi être lancé seul pour une config unique.
3. **`collect_results.py`** — agrège tous les résultats produits en `results_dev.csv` et `results_eval.csv`.
