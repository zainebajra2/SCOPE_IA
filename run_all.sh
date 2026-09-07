#!/usr/bin/env bash
# Regenere toutes les configurations rapportees.
# Les resultats dev et eval sont ecrits dans des dossiers SEPARES : un run eval
# ne peut plus ecraser un run dev de configuration voisine.
#
#   bash run_all.sh
set -u

ROOT_DEV=/mnt/storage_1_10T/scope/data/DCASE2026_Task2/dev
ROOT_EVAL=/mnt/storage_1_10T/scope/data/DCASE2026_Task2/eval
CACHE=/mnt/storage_1_10T/scope/cache/feat_cache
EVALUATOR=$HOME/dcase2026/dcase2026_task2_evaluator
GPU=0

run_dev () {
  echo ">>> dev : $*"
  CUDA_VISIBLE_DEVICES=$GPU python -u asd_pipeline.py \
      --data-root "$ROOT_DEV" --cache-dir "$CACHE" --out asd_results_dev "$@" 2>&1 \
      | grep -E "SCORE OFFICIEL|Resultats dans|Traceback|Error|echoue|epoch|pseudo-classes|grille de tokens"
}

run_eval () {
  local name=$1; shift
  echo ">>> eval [$name] : $*"
  CUDA_VISIBLE_DEVICES=$GPU python -u asd_pipeline.py \
      --data-root "$ROOT_EVAL" --cache-dir "$CACHE" --out asd_results_eval \
      --export-scores "$@" 2>&1 \
      | grep -E "SCORE OFFICIEL|Resultats dans|Traceback|Error|echoue|epoch|pseudo-classes|grille de tokens"
  mkdir -p "$EVALUATOR/teams/SCOPE/$name"
  cp asd_results_eval/dcase_submission/*.csv "$EVALUATOR/teams/SCOPE/$name/"
}

BASE="--frontend ast --mask none"
POOL="--pooling freq --band-agg max"

echo "===== DEV : choix du pooling (scoring standard, sans per-domain) ====="
for P in mean meanstd freq freq_meanstd; do
  run_dev $BASE --pooling $P
done
run_dev $BASE $POOL

echo "===== DEV : scoring ====="
run_dev $BASE $POOL --domain-align
run_dev $BASE $POOL --per-domain
run_dev $BASE $POOL --per-domain --knn 2
run_dev $BASE $POOL --per-domain --knn 8

echo "===== DEV : masque inter-canaux (resultat negatif) ====="
run_dev --frontend ast --mask leveldiff --pooling mean
run_dev --frontend ast --mask leveldiff --pooling mean --alpha 0.5
run_dev --frontend ast --mask leveldiff $POOL --per-domain
run_dev --frontend ast --mask leveldiff $POOL --per-domain --swap-channels

echo "===== DEV : FarMix (resultat negatif) ====="
for FM in 1 2 4; do
  run_dev $BASE $POOL --per-domain --farmix $FM
done
for R in "-5 5" "0 10" "10 30"; do
  set -- $R
  run_dev $BASE $POOL --per-domain --farmix 2 --farmix-snr-min $1 --farmix-snr-max $2
done

echo "===== DEV : calibration du seuil ====="
for TH in otsu mixture; do
  run_dev $BASE $POOL --per-domain --threshold $TH
done
for Q in 0.4 0.5 0.6 0.7 0.8 0.9 0.95; do
  run_dev $BASE $POOL --per-domain --threshold gamma --threshold-q $Q
done

echo "===== DEV : graines (deterministe cote AST, variable cote CNN) ====="
for S in 1 2; do
  run_dev $BASE $POOL --per-domain --seed $S
done

echo "===== DEV : CNN entraine de zero (resultat negatif) ====="
run_dev --frontend cnn --mask none $POOL --per-domain --pseudo-labels 16
run_dev --frontend cnn --mask none $POOL --per-domain --pseudo-labels 0
run_dev --frontend cnn --mask none $POOL --per-domain --pseudo-labels 16 --seed 1

echo "===== EVAL ====="
run_eval align       $BASE $POOL --domain-align
run_eval noalign     $BASE $POOL
run_eval perdom      $BASE $POOL --per-domain
run_eval perdom_fm2  $BASE $POOL --per-domain --farmix 2 --farmix-snr-min 0 --farmix-snr-max 10
run_eval thr_gamma07 $BASE $POOL --per-domain --threshold gamma --threshold-q 0.7
run_eval thr_otsu    $BASE $POOL --per-domain --threshold otsu
run_eval leveldiff   --frontend ast --mask leveldiff $POOL --per-domain

echo "===== Evaluateur officiel ====="
( cd "$EVALUATOR" && bash 03_evaluation_eval_data.sh )

echo "===== Termine : python collect_results.py --results asd_results_dev ====="