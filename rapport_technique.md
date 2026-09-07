# Détection non supervisée d'anomalies sonores sur enregistrements à deux canaux

## Exploitation du contraste proche/lointain sans simulation ni données externes

Projet SCOPE — rapport technique interne, préparation d'une communication en traitement du signal audio.

---

### Conventions de lecture

Ce rapport sépare trois statuts d'information, signalés explicitement.

| Statut | Signification |
|---|---|
| **Mesure** | Valeur lue dans un fichier du dépôt ou recalculée depuis les scores bruts. Chiffre reproductible. |
| **Chiffre publié** | Valeur rapportée par un tiers (organisateurs du challenge, équipe MERL). Aucun recalcul de notre part. Jamais agrégée avec nos mesures. |
| **Hypothèse** | Explication mécaniste proposée. Non établie par les mesures présentées. |

Toutes les mesures proviennent de `results_dev.csv`, `results_eval.csv`, `asd_results_dev/`, `asd_results_eval/`, `dcase2026_task2_evaluator/teams_result/`, `e0_results/e0b_summary.csv` et `asd_pipeline.py`. Les hyperparamètres sont lus dans `asd_pipeline.py`. Les effectifs de données sont comptés directement sur les fichiers audio.

---

## 1. Contexte et objectif

### 1.1 Le challenge

DCASE 2026 Challenge Task 2, « Noise-aware Unsupervised Anomalous Sound Detection for Machine Condition Monitoring ». Le présent travail ne visait pas la soumission au challenge, clos, mais une publication en conférence avec comparaison aux résultats publiés.

La nouveauté 2026 est la mise à disposition d'enregistrements à **deux canaux**, captés à des distances différentes de la machine. Le canal 1 est proche, le canal 2 est éloigné. Les organisateurs présentent l'écart de rapport signal sur bruit et l'écart de caractéristiques spectrales entre les deux canaux comme un indice exploitable pour séparer la machine du bruit de fond, collectable sans arrêter la production.

Les contraintes du challenge sont les suivantes.

- Entraînement non supervisé sur clips normaux uniquement.
- Généralisation entre un domaine source riche et un domaine cible pauvre.
- Machines du jeu d'évaluation totalement absentes du jeu de développement, contrainte *first-shot*.
- Attributs de fonctionnement parfois absents.

### 1.2 Chiffres publiés servant de référence

**Chiffres publiés.** Les valeurs suivantes sont rapportées par des tiers. Elles ne sont pas recalculées ici et ne sont jamais fusionnées avec nos mesures.

| Système | Développement | Évaluation |
|---|---|---|
| Baseline autoencodeur MSE | 56.66 | 59.80 |
| Baseline Mahalanobis | 57.66 | 54.76 |
| BEATs gelé | 60.28 | — |
| NA-BEATs | 63.18 | — |
| Meilleur système MERL | 66.20 | — |
| Vainqueur du challenge | — | 70.24 |

Les deux baselines officielles **inversent leur classement** entre les deux jeux. Mahalanobis dépasse MSE d'un point sur le développement, et lui est inférieur de cinq points sur l'évaluation. Ce point est repris en section 7 : il indique que l'instabilité du transfert développement → évaluation n'est pas propre à notre système.

### 1.3 Objectif du travail

Obtenir un bénéfice comparable à celui du système gagnant sans simulation acoustique, sans données externes et sans entraînement de représentation, avec un système léger et déployable.

---

## 2. Problématique et positionnement

### 2.1 Le système gagnant

**Chiffres publiés et description publiée.** Fujimura et al. (MERL) proposent un *noise-aware self-supervised learning*. Des couches d'attention croisée entraînables sont insérées dans un BEATs/EAT gelé. La représentation du canal proche est raffinée par celle du canal lointain. L'entraînement se fait par distillation contre la représentation du signal propre, sur données **simulées** avec Pyroomacoustics, en utilisant FSD50K comme source de signaux et WHAM!, DEMAND et QUT-NOISE comme banques de bruit externes. S'y ajoutent un fine-tuning discriminatif avec LoRA et sub-cluster AdaCos, des pseudo-étiquettes k-means, une *memory bank* par bande de fréquence, un *relative deviation pooling* et un ensemble de front-ends.

Leur ablation publiée, sur le développement, à backend identique :

| Machine | BEATs gelé | NA-BEATs | Δ |
|---|---|---|---|
| bearingEmu | 62.94 | 62.48 | −0.5 |
| fan | 52.12 | 53.15 | +1.0 |
| gearboxEmu | 64.43 | 64.43 | 0.0 |
| sliderEmu | 63.57 | 65.77 | +2.2 |
| ToyCar | 57.40 | 57.26 | −0.1 |
| ToyCarEmu | 57.62 | 58.57 | +1.0 |
| valveEmu | 66.59 | 93.34 | +26.8 |
| **Total** | **60.28** | **63.18** | **+2.9** |

Le gain agrégé de 2.9 points est concentré sur une seule machine. La valve apporte +26.8 ; les six autres machines totalisent entre −0.5 et +2.2.

### 2.2 Hypothèse sur l'origine du gain

**Hypothèse.** Pour un transitoire, l'écart entre le canal proche et le canal lointain porte de l'information : le front d'attaque arrive avec un rapport signal sur bruit très différent selon la distance, et la queue réverbérée diffère. Pour une source stationnaire large bande, spectralement proche du bruit d'usine, un objectif de débruitage instantané n'a presque rien à exploiter : les deux canaux voient le même mélange, à un gain près.

### 2.3 Ce que nos mesures disent de cette hypothèse

**Mesure.** `e0_results/e0b_summary.csv` classe chaque machine en régime impulsif ou stationnaire, sur critère de modulation temporelle (seuil `MOD_MIN_DB = 6.0` dB, lu dans `e0_coherence_analysis.py`). L'analyse porte sur **60 clips** du split train source par machine, tirés au hasard, et non sur le jeu complet.

| Machine | Modulation (dB) | Régime | Rapport Q1 sur raies (dB) | Δ NA-BEATs (publié) |
|---|---|---|---|---|
| ToyCar | 8.84 | impulsif | +19.53 | −0.1 |
| ToyCarEmu | 8.54 | impulsif | +11.22 | +1.0 |
| valveEmu | 8.77 | impulsif | −3.60 | +26.8 |
| bearingEmu | 4.36 | stationnaire | +0.58 | −0.5 |
| sliderEmu | 4.04 | stationnaire | −0.27 | +2.2 |
| fan | 2.62 | stationnaire | −3.19 | +1.0 |
| gearboxEmu | 1.94 | stationnaire | −2.83 | 0.0 |

**Ce que la mesure établit.** valveEmu, la seule machine où NA-BEATs apporte un gain important, est bien classée impulsive par nos mesures, avec la deuxième modulation la plus forte des sept.

**Ce que la mesure n'établit pas.** La colonne de régime ne sépare pas les machines qui gagnent de celles qui ne gagnent pas. Les deux autres machines impulsives, ToyCar et ToyCarEmu, obtiennent −0.1 et +1.0. Le deuxième gain le plus élevé, sliderEmu à +2.2, porte sur une machine stationnaire. Le régime est donc **compatible** avec l'hypothèse 2.2 ; il ne la démontre pas. Toute affirmation plus forte demanderait une analyse par machine du contenu transitoire corrélée aux gains, sur le jeu complet.

### 2.4 Positionnement

Notre système renonce à trois éléments du système gagnant : la simulation acoustique, les banques de bruit externes, et tout entraînement de représentation. Il conserve deux idées : le raisonnement par bande de fréquence, et l'exploitation du second canal. Les deux voies d'exploitation du second canal testées ici sont rejetées par les mesures (section 6). Ce qui subsiste et fonctionne relève du *scoring* et non de la représentation (section 5).

---

## 3. Données et protocole d'évaluation

### 3.1 Effectifs

**Mesure**, comptée directement sur les fichiers audio des deux jeux.

| Jeu | Machines | Train par machine | Test par machine |
|---|---|---|---|
| Développement | 7 | 1000 (990 source, 10 cible) | 200 |
| Évaluation | 5 | 1000 (990 source, 10 cible) | 200 |

Machines du développement : ToyCar, ToyCarEmu, bearingEmu, fan, gearboxEmu, sliderEmu, valveEmu.
Machines de l'évaluation : BlowerDustCollector, Sander, SewingMachine, ToothBrush, ToyDrone. Aucune n'apparaît dans le développement.

Le déséquilibre du train est de **990 clips source contre 10 clips cible**, identique pour les douze machines. Ce rapport de 99 pour 1 est le point de départ de la contribution de la section 5.2.

**Mesure.** Le test du développement est équilibré exactement, pour les sept machines :

| Composition du test (par machine) | Effectif |
|---|---|
| Total | 200 |
| normal | 100 |
| anomaly | 100 |
| source | 100 |
| cible | 100 |
| source × normal | 50 |
| source × anomaly | 50 |
| cible × normal | 50 |
| cible × anomaly | 50 |

Vérifié sur les 32 fichiers `scores_*.npz` de `asd_results_dev/` : une seule signature d'effectifs, aucune exception.

Sur l'évaluation, les labels ne figurent pas dans les noms de fichiers. Les 200 clips de test par machine sont donc de domaine et de label inconnus côté pipeline.

### 3.2 Format audio

**Mesure**, lue dans les en-têtes WAV.

| Propriété | Valeur |
|---|---|
| Fréquence d'échantillonnage | 16 kHz |
| Canaux | 2 |
| Quantification | 16 bits, entier signé |
| Durée | 10 s (bearingEmu, fan, gearboxEmu, sliderEmu, valveEmu), 11 s (ToyCar), 12 s (ToyCarEmu) |

La durée n'est pas uniforme. Cinq machines sur sept font 10 s ; ToyCar fait 11 s et ToyCarEmu 12 s. Le front-end en tient compte par un découpage en fenêtres (section 4.2).

### 3.3 Attributs

**Mesure**, comptée sur les noms de fichiers du split train.

| Machine | Valeurs d'attribut distinctes |
|---|---|
| ToyCarEmu | 22 |
| gearboxEmu | 9 |
| fan | 3 |
| ToyCar, bearingEmu, sliderEmu, valveEmu | 1 (`noAttributes`) |

Quatre machines sur sept ne portent aucun attribut. Le split test de gearboxEmu contient 10 valeurs distinctes, soit une de plus que le train. Cette asymétrie n'affecte pas la branche AST, qui n'utilise pas les attributs.

### 3.4 Le dossier `supplemental`

**Mesure.** Le parcours de l'arborescence par `discover()` ne trouve aucun split `supplemental`, ni sur le développement ni sur l'évaluation, alors que `asd_pipeline.py` le déclare dans `SPLITS` et sait le lire.

Ce dossier, décrit par la fiche Zenodo, contient des enregistrements de machine seule et de bruit seul. Son absence dans l'extraction utilisée empêche de mesurer directement la contamination du canal 2 par la machine. Cette limite est reprise en section 8.

### 3.5 Métriques

Le score officiel est la **moyenne harmonique** de toutes les AUC par domaine et de toutes les pAUC, sur toutes les machines. Pour le développement, cela fait 7 machines × 3 valeurs = 21 valeurs. Pour l'évaluation, 5 machines × (2 AUC + 1 pAUC) = 15 valeurs.

La pAUC est l'AUC partielle sur la région de faible taux de faux positifs, `FPR ≤ 0.1`, standardisée selon McClish pour qu'un système aléatoire donne 0.5. L'implémentation délègue à `sklearn.metrics.roc_auc_score(..., max_fpr=0.1)`, ce qu'utilisent les baselines officielles.

**Mesure — vérification indépendante.** Le score officiel a été recalculé depuis les fichiers `scores_*.npz`, par un script écrit sans réutiliser `collect_results.py`. Trois configurations nommées, puis les 32 configurations du développement.

| Configuration | Recalcul | `results_dev.csv` | `summary_*.json` |
|---|---|---|---|
| `ast none freq/max --per-domain` | 57.91 | 57.91 | 57.91 |
| `ast none freq/max` | 58.56 | 58.56 | 58.56 |
| `ast leveldiff freq/max --per-domain` | 58.35 | 58.35 | 58.35 |
| `ast none freq/max --per-domain --farmix 2` (0–10 dB) | 58.80 | 58.80 | 58.80 |
| `cnn --pseudo-labels 16 --seed 1` | 49.04 | 49.04 | 49.04 |
| `cnn --pseudo-labels 0 --seed 0` | 50.21 | 50.21 | 50.21 |

Écart maximal sur les 32 configurations : **0.0047 point**, imputable au seul arrondi à deux décimales. Le détail machine par machine coïncide au centième.

**Mesure.** Sur l'évaluation, la relation entre le score officiel et les moyennes harmoniques partielles produites par l'évaluateur officiel est vérifiée pour les sept soumissions :

`official = 15 / (10 / hm_AUC + 5 / hm_pAUC)`

Écart maximal 0.005 point. Le score officiel est également retrouvé en recalculant la moyenne harmonique des 10 AUC et 5 pAUC par machine lues dans `results_eval.csv`.

### 3.6 Résolution de la pAUC

La région `FPR ≤ 0.1` ne concerne que les clips normaux les mieux classés. Avec 100 clips normaux par machine, elle en couvre **10**.

**Mesure — borne.** Déplacer un seul clip normal modifie l'aire partielle brute de au plus `1/100 = 0.01`. Après standardisation McClish, dont le facteur est `0.5 / (0.1 − 0.1²/2) = 0.5 / 0.095`, cela borne la variation de pAUC standardisée à :

`0.5 × 0.01 / 0.095 = 5.26 points`

**Mesure — amplitude observée.** Sur ToyCar, dans la configuration retenue, chacun des 100 clips normaux a été déplacé successivement aux deux extrêmes de l'échelle de score, soit 200 essais. La pAUC de référence vaut 51.68. Les écarts observés vont de **−0.74 à +0.95 point**.

Les 5.26 points sont donc une **borne supérieure**, atteinte seulement si le clip déplacé est celui qui gouverne toute la région. Sur des distributions de scores réelles, l'effet mesuré reste sous le point. Les deux chiffres doivent être cités ensemble : la borne dit que la métrique est mal résolue en principe, la mesure dit de combien elle bouge en pratique.

Cette borne suffit à écarter toute interprétation d'un écart de pAUC inférieur au point comme un effet de système.

---

## 4. Système proposé

Tous les éléments de cette section sont lus dans `asd_pipeline.py`.

### 4.1 Vue d'ensemble

Le système ne comporte aucun entraînement dans sa configuration retenue. Il enchaîne quatre étapes.

1. Calcul d'un mel-spectrogramme sur le canal 1, éventuellement pondéré par un masque calculé à partir des deux canaux.
2. Extraction d'embeddings par un Audio Spectrogram Transformer pré-entraîné et gelé.
3. Score d'anomalie par distance cosinus aux plus proches voisins parmi les clips normaux d'entraînement, bande de fréquence par bande de fréquence.
4. Calibration d'un seuil de décision sans étiquette.

### 4.2 Front-end

| Élément | Valeur |
|---|---|
| Checkpoint | `MIT/ast-finetuned-audioset-10-10-0.4593` |
| Paramètres | 86 187 264, tous gelés (`requires_grad_(False)`) |
| Entrée | fbank Kaldi, 128 bandes mel, fenêtre 25 ms, pas 10 ms, `htk_compat=True`, `dither=0.0`, sans énergie |
| Normalisation | `(log M − AST_MEAN) / (2 × AST_STD)`, avec `AST_MEAN = −4.2677393`, `AST_STD = 4.5689974` |
| Longueur de fenêtre | 1024 trames (`AST_FRAMES`) |
| Remplissage | valeur constante `(0 − AST_MEAN) / (2 × AST_STD)`, reproduisant le comportement d'AST sur fbank brut |

Les clips faisant 10 à 12 s, soit 1000 à 1200 trames, une troncature à 1024 trames perdrait la fin des plus longs. Le front-end découpe donc chaque clip en fenêtres de 1024 trames et **moyenne les embeddings des fenêtres**.

L'entrée est fournie directement en `input_values`, sans passer par le feature extractor de la bibliothèque. Ce choix est nécessaire : le masque est appliqué sur le mel, il ne faut pas recalculer le mel depuis la forme d'onde.

### 4.3 Pooling des tokens

Les tokens de patch d'AST forment une grille bande de fréquence × position temporelle. Ses dimensions se déduisent de la configuration du checkpoint par les formules du code :

```
f_dim = (num_mel_bins − patch_size) / frequency_stride + 1 = (128 − 16) / 10 + 1 = 12
t_dim = (max_length   − patch_size) / time_stride      + 1 = (1024 − 16) / 10 + 1 = 101
```

soit **12 bandes × 101 positions**, 1212 tokens de patch, plus 2 tokens spéciaux (CLS et distillation) écartés du pooling fréquentiel.

Quatre stratégies sont implémentées.

| `--pooling` | Sortie | Description |
|---|---|---|
| `mean` | (N, 1, d) | moyenne de tous les tokens, axe fréquentiel écrasé |
| `meanstd` | (N, 1, 2d) | moyenne et écart-type sur tous les tokens |
| `freq` | (N, 12, d) | moyenne sur l'axe temporel seul, 12 bandes conservées |
| `freq_meanstd` | (N, 12, 2d) | moyenne et écart-type sur l'axe temporel, 12 bandes conservées |

### 4.4 Score par plus proches voisins

Le score est la distance cosinus moyenne aux `k` plus proches voisins parmi les clips normaux d'entraînement, calculée **indépendamment dans chaque bande** puis agrégée entre bandes par `--band-agg` (`mean` ou `max`).

Quand la banque est interrogée par elle-même, pour estimer l'échelle ou le score de référence, la distance du clip à lui-même vaut zéro par construction. Le paramètre `skip_self` l'écarte au lieu de la moyenner. Sans cela le score de référence est sous-estimé et tout seuil calibré dessus est trop bas.

### 4.5 Normalisation par domaine (`--per-domain`)

**Mesure du problème.** Avec 990 références source contre 10 cible, un clip du domaine cible est presque toujours apparié à une référence source. La distance mesurée reflète alors l'écart entre domaines plutôt que la présence d'une anomalie.

**Solution implémentée.** La banque est partitionnée par domaine. Chaque partie est interrogée séparément. Chaque distance est divisée par l'échelle propre de sa partie, estimée comme la **médiane des distances des références de cette partie entre elles**, en laissant-un-de-côté. Le score final est le minimum des deux distances normalisées.

```
pour chaque banque b :
    échelle_b = médiane_références( kNN(b, b, skip_self=True) )
    D_b       = kNN(b, Q) / échelle_b
D = min(D_source, D_cible)
```

Prendre le minimum brut sans normaliser ne servirait à rien : le minimum sur une union vaut le minimum des minimums. La normalisation est l'élément actif.

**Point important pour l'applicabilité.** Aucune étiquette de domaine des clips de **test** n'est utilisée. Seule la banque d'entraînement est partitionnée, et son domaine est connu par construction. La méthode s'applique donc telle quelle à l'évaluation, où le domaine des clips de test est inconnu.

### 4.6 Variante `--domain-align`

Alternative testée : recentrer les références du domaine cible sur la moyenne du domaine source, puis renormaliser en norme L2. Incompatible avec `--per-domain`, qui traite le problème autrement. Les mesures la rejettent (section 6.4).

### 4.7 Masque inter-canaux (`--mask leveldiff`)

Le rapport de puissance entre les deux canaux vaut `1 + SNR`, mais seulement après correction de l'écart de gain entre les deux microphones. Le gain de Wiener correspondant s'écrit :

```
ratio = M1 / M2
c     = percentile(ratio, calib_pct)
G     = clip(1 − β · c / ratio, gfloor, 1)^α
```

Le facteur `c` est estimé par clip, sur un percentile bas du rapport `M1/M2`, c'est-à-dire dans les bandes où la machine est absente. Dans les bandes de bruit le rapport vaut `c` et `G` tombe au plancher ; dans les bandes où la machine émet, le rapport dépasse `c` et `G` remonte vers 1. Aucune étiquette et aucune mesure préalable ne sont requises.

Le masque est appliqué au mel de puissance avant normalisation et avant le modèle.

### 4.8 FarMix (`--farmix`)

Augmentation de la banque de référence. Chaque clip normal reçoit du bruit prélevé sur le **canal 2 d'un autre clip**, à un rapport signal sur bruit tiré uniformément dans `[--farmix-snr-min, --farmix-snr-max]`. Le mélange se fait dans le domaine mel-puissance, où les puissances de deux signaux décorrélés s'additionnent.

Un décalage circulaire de permutation garantit qu'aucun clip ne reçoit son propre canal 2, ce qui réinjecterait la signature de la machine au lieu du bruit.

`--farmix N` ajoute `N` copies augmentées à la banque, qui compte alors `(N+1) × 1000` références par machine.

### 4.9 Calibration du seuil (`choose_threshold`)

Quatre stratégies non supervisées.

| `--threshold` | Données utilisées | Description |
|---|---|---|
| `gamma` | scores des clips normaux du train | loi gamma ajustée, quantile `q`. Procédure de la baseline officielle. |
| `percentile` | scores des clips normaux du train | quantile `q` empirique, sans hypothèse de forme |
| `otsu` | scores de test, échelle log | seuil maximisant la variance inter-classes, 128 classes d'histogramme |
| `mixture` | scores de test, échelle log | mélange de deux gaussiennes, seuil au croisement des composantes |

L'ajustement gamma essaie `floc = 0` avant `floc = min(scores) − 1e-6`. Ce point est développé en section 7.4 : sans cet essai, l'ajustement échouait systématiquement en configuration `--per-domain`.

Les stratégies `otsu` et `mixture` exploitent la distribution des scores de test, donc un accès par lot aux données. Aucune étiquette n'est utilisée, mais cela s'écarte d'un fonctionnement en ligne. Limite déclarée en section 8.

### 4.10 Branche CNN

Branche alternative, entraînée de zéro, conservée comme résultat négatif (section 6.3).

| Élément | Valeur |
|---|---|
| Architecture | 4 blocs de 2 convolutions 3×3 + BatchNorm + ReLU, MaxPool 2×2, canaux 1→32→64→128→256 |
| Tête | pooling moyen adaptatif, projection linéaire vers `--emb-dim 128`, ArcFace |
| ArcFace | échelle 30.0, marge 0.7 |
| Paramètres | 1 210 400 sans pseudo-étiquettes, 1 218 080 avec 16 pseudo-classes |
| Classes | machine × attributs, un modèle unique pour toutes les machines |
| Optimiseur | AdamW, `lr = 1e-3`, `weight_decay = 1e-4`, cosine annealing |
| Époques | 40, batch 64 |
| Crop | 128 trames, position aléatoire |
| SpecAugment | 2 passes, jusqu'à 15 % de l'axe fréquentiel et de l'axe temporel |
| Lissage d'étiquettes | 0.1 |
| Pseudo-étiquettes | k-means à 16 classes sur le profil spectral moyen, appliqué aux machines à attribut unique |
| Déterminisme | `cudnn.deterministic = True`, `cudnn.benchmark = False`, `use_deterministic_algorithms(True, warn_only=True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` |

**Mesure.** Le nombre de classes effectives dépend des pseudo-étiquettes, pour 7000 clips d'entraînement dans les deux cas.

| Configuration | Classes | Composition |
|---|---|---|
| `--pseudo-labels 0` | **38** | 22 (ToyCarEmu) + 9 (gearboxEmu) + 3 (fan) + 4 × 1 (machines sans attribut) |
| `--pseudo-labels 16` | **98** | les 34 attributs réels + 4 × 16 pseudo-classes |

Les pseudo-étiquettes s'appliquent aux quatre machines à attribut unique — ToyCar, bearingEmu, sliderEmu, valveEmu — avec des tailles de classe très déséquilibrées : 3 à 98 clips pour ToyCar, 8 à 210 pour bearingEmu, 9 à 110 pour sliderEmu, 2 à 131 pour valveEmu. Sans pseudo-étiquettes, ces quatre machines forment quatre classes de 1000 clips chacune.

Le mode déterministe est activé par défaut et désactivable par `--no-deterministic`. Il concerne la seule branche CNN : la branche AST n'entraîne rien et ne comporte aucune convolution apprise.

### 4.11 Hyperparamètres du mel non-AST et valeurs par défaut

| Paramètre | Valeur par défaut |
|---|---|
| `N_FFT`, `HOP`, `N_MELS`, `FMIN` | 1024, 512, 128, 20.0 Hz |
| `--alpha` (exposant du masque) | 1.0 |
| `--beta` (sur-soustraction) | 1.0 |
| `--calib-pct` (percentile de calibration) | 20.0 |
| `--gfloor` (plancher de gain) | 0.05, soit −13 dB |
| `--knn` | 1 |
| `--band-agg` | `mean` |
| `--pooling` | `mean` |
| `--threshold`, `--threshold-q` | `gamma`, 0.9 |
| `--farmix`, plage SNR | 0, 0.0–20.0 dB |
| `--seed` | 0 |
| Mode déterministe | activé (`--no-deterministic` pour désactiver) |

### 4.12 Configuration retenue

| Paramètre | Valeur |
|---|---|
| Front-end | AST gelé, 86.19 M paramètres |
| Masque | aucun |
| Pooling | `freq`, 12 bandes |
| Agrégation entre bandes | `max` |
| Voisinage | `--knn 1` |
| Normalisation | `--per-domain` |
| FarMix | désactivé |

**Mesure.** Aucun entraînement. Inférence 115.0 ms/clip pour cette configuration, 114.0 ms/clip sans `--per-domain`. À titre de comparaison, la branche CNN entraîne en 131 s et infère en 2.13 ms/clip.

Le choix de cette configuration, qui n'est pas la meilleure sur le développement, est justifié en section 7.5.

### 4.13 Traçabilité des résultats

**Mesure.** `asd_results_dev/` contient 32 tags uniques et 96 fichiers, soit 32 triplets `scores`/`metrics`/`summary` complets. Aucun triplet incomplet, aucun tag dupliqué. `asd_results_eval/` contient 7 tags uniques et 21 fichiers, plus un dossier de transit `dcase_submission/`.

Le tag de résultat encode tous les paramètres discriminants, y compris le nom du jeu de données, `--knn`, `--domain-align`, le plancher de gain et la plage SNR de FarMix. Les résultats du développement et de l'évaluation sont écrits dans deux dossiers séparés. Une collision de tag entre deux configurations distinctes, ou entre un run de développement et un run d'évaluation, n'est plus possible.

Les fichiers `scores_*.npz` du jeu d'évaluation ne contiennent aucun score, le jeu étant non labellisé : 22 octets chacun. `collect_results.py` les ignore sur critère de taille.

---

## 5. Résultats

### 5.1 Pooling fréquentiel

**Mesure**, développement, sans `--per-domain`, sans masque.

| Pooling | Agrégation | Score officiel | Δ vs `mean` |
|---|---|---|---|
| `mean` | — | 56.54 | référence |
| `meanstd` | — | 56.89 | +0.35 |
| `freq` | `mean` | 58.09 | +1.55 |
| `freq_meanstd` | `mean` | 58.33 | +1.79 |
| `freq` | `max` | **58.56** | **+2.02** |

Conserver les 12 bandes séparées et scorer bande par bande apporte 1.55 point avec agrégation par moyenne, 2.02 points avec agrégation par maximum.

**Mesure.** L'agrégation par maximum dépasse la moyenne de **+0.47** point, à pooling identique.

**Hypothèse.** La signature d'un défaut mécanique occupe une bande de fréquence précise. Moyenner tous les tokens dilue cette signature dans les bandes intactes. L'agrégation par maximum est sensible aux anomalies très localisées ; l'agrégation par moyenne est plus robuste au bruit. Les mesures sur ce jeu favorisent la sensibilité.

### 5.2 Normalisation par domaine

C'est le résultat central du rapport.

**Mesure.** Scores officiels, à pooling et agrégation identiques (`freq` / `max`), sans masque.

| Configuration | Développement | Évaluation | Écart eval − dev |
|---|---|---|---|
| Aucune correction (`noalign`) | 58.56 | 52.28 | — |
| `--domain-align` (`align`) | 58.29 | 49.96 | — |
| `--per-domain` (`perdom`) | 57.91 | **56.75** | — |
| **Δ `per-domain` − aucune** | **−0.65** | **+4.47** | **+5.12** |

`--per-domain` coûte 0.65 point sur le développement et en rapporte 4.47 sur l'évaluation.

**Mesure — décomposition par domaine.** L'effet se lit dans la répartition entre AUC source et AUC cible.

| Jeu | Configuration | hm AUC source | hm AUC cible | hm pAUC |
|---|---|---|---|---|
| Développement | `noalign` | 64.01 | 59.47 | 53.20 |
| Développement | `per-domain` | 61.52 | 61.00 | 52.20 |
| Développement | Δ | −2.49 | +1.53 | −1.00 |
| Évaluation | `noalign` | 60.87 | 45.36 | 52.88 |
| Évaluation | `per-domain` | 59.98 | 56.00 | 54.56 |
| Évaluation | Δ | −0.89 | **+10.64** | +1.68 |

La méthode échange de l'AUC source contre de l'AUC cible. Sur le développement l'échange est presque neutre, avec un léger déficit. Sur l'évaluation il est très favorable : 10.64 points d'AUC cible pour 0.89 point d'AUC source.

**Mesure — détail par machine, évaluation.**

| Machine | AUC source `noalign` | AUC source `perdom` | Δ | AUC cible `noalign` | AUC cible `perdom` | Δ |
|---|---|---|---|---|---|---|
| BlowerDustCollector | 51.86 | 61.42 | +9.56 | 60.72 | 69.00 | +8.28 |
| Sander | 54.86 | 55.70 | +0.84 | 46.40 | 51.92 | +5.52 |
| SewingMachine | 59.46 | 56.84 | −2.62 | 58.26 | 57.16 | −1.10 |
| ToothBrush | 65.60 | 59.98 | −5.62 | 52.40 | 67.50 | **+15.10** |
| ToyDrone | 79.52 | 67.26 | −12.26 | 27.80 | 43.06 | **+15.26** |

Deux machines concentrent le gain. ToyDrone et ToothBrush gagnent plus de 15 points d'AUC cible chacune. ToyDrone perd 12.26 points d'AUC source en échange ; la moyenne harmonique, qui pénalise les valeurs basses, favorise cet échange.

**Mesure — le cas ToyDrone.** Avec `--domain-align`, l'AUC cible de ToyDrone tombe à **21.48**, soit très en dessous du hasard. `noalign` donne 27.80. `--per-domain` remonte à 43.06.

| Configuration | ToyDrone AUC source | ToyDrone AUC cible |
|---|---|---|
| `align` | 80.88 | 21.48 |
| `noalign` | 79.52 | 27.80 |
| `perdom` | 67.26 | 43.06 |
| `leveldiff` (per-domain) | 67.62 | 46.04 |

**Hypothèse.** Une AUC cible sous 25 avec une AUC source à 80 est la signature d'une détection inversée sur le domaine cible : plus un clip est typiquement cible, plus il est jugé anormal. Le score mesure alors l'écart entre domaines et non l'anomalie. `--per-domain` supprime cet effet sans le corriger complètement : 43.06 reste sous le hasard.

**Mesure — détail par machine, développement.**

| Machine | AUC src `noalign` | AUC src `perdom` | AUC cible `noalign` | AUC cible `perdom` | pAUC `noalign` | pAUC `perdom` |
|---|---|---|---|---|---|---|
| ToyCar | 68.52 | 68.72 | 68.08 | 67.96 | 53.16 | 51.68 |
| ToyCarEmu | 60.48 | 58.96 | 77.44 | 71.44 | 52.32 | 49.16 |
| bearingEmu | 60.28 | 63.80 | 58.84 | 59.80 | 57.53 | 59.05 |
| fan | 52.76 | 50.20 | 46.28 | 47.24 | 49.79 | 48.68 |
| gearboxEmu | 68.44 | 59.40 | 54.32 | 59.40 | 53.95 | 51.58 |
| sliderEmu | 66.04 | 57.88 | 57.68 | 59.12 | 52.84 | 52.74 |
| valveEmu | 77.36 | 79.56 | 63.48 | 69.40 | 53.42 | 53.84 |

Sur le développement, gearboxEmu et sliderEmu perdent 9.04 et 8.16 points d'AUC source, pour respectivement +5.08 et +1.44 d'AUC cible. L'échange y est défavorable, ce qui explique le coût de 0.65 point.

### 5.3 Résultat de l'évaluation

**Mesure**, lue dans `results_eval.csv`, produite par l'évaluateur officiel `nttcslab/dcase2026_task2_evaluator` sur les sept soumissions.

| Soumission | Score officiel | hm AUC | hm pAUC | AUC source | AUC cible | hm F1 |
|---|---|---|---|---|---|---|
| `perdom` | **56.75** | 57.92 | 54.56 | 59.98 | 56.00 | 34.89 |
| `thr_gamma07` | **56.75** | 57.92 | 54.56 | 59.98 | 56.00 | 51.78 |
| `thr_otsu` | **56.75** | 57.92 | 54.56 | 59.98 | 56.00 | 41.65 |
| `leveldiff` | 56.67 | 58.16 | 53.91 | 60.36 | 56.11 | 36.47 |
| `perdom_fm2` | 54.40 | 54.60 | 54.00 | 60.98 | 49.43 | 47.89 |
| `noalign` | 52.28 | 51.98 | 52.88 | 60.87 | 45.36 | 34.55 |
| `align` | 49.96 | 48.81 | 52.42 | 60.83 | 40.75 | 34.78 |

Les trois soumissions `perdom`, `thr_gamma07` et `thr_otsu` partagent le même score officiel de 56.75. C'est attendu : elles diffèrent seulement par la stratégie de seuillage, et le seuil n'entre ni dans l'AUC ni dans la pAUC. Elles diffèrent en revanche par le F1.

**Mesure — comparaison aux chiffres publiés.** Notre meilleur score d'évaluation est 56.75. Le chiffre publié pour la baseline autoencodeur MSE est 59.80, celui du vainqueur 70.24. Notre système reste **3.05 points sous la baseline MSE** et 13.49 points sous le vainqueur. Le chiffre publié pour la baseline Mahalanobis est 54.76 ; notre système la dépasse de 1.99 point. Ces comparaisons croisent une mesure et des chiffres publiés ; elles ne sont pas une mesure.

### 5.4 Résultat du développement

**Mesure.** Les six meilleures configurations du développement.

| Rang | Score | Configuration |
|---|---|---|
| 1 | 58.80 | `per-domain` + FarMix 2, SNR 0–10 dB |
| 2 | 58.69 | `per-domain` + FarMix 4, SNR 0–20 dB |
| 3 | 58.56 | `freq`/`max`, aucune correction de domaine |
| 4 | 58.36 | `per-domain` + FarMix 1, SNR 0–20 dB |
| 5 | 58.35 | `per-domain` + masque `leveldiff` |
| 6 | 58.33 | `per-domain` + FarMix 2, SNR 0–20 dB |

Étendue des six premières : **0.47 point**. La chaîne étant déterministe (section 7.1), ces écarts sont réels ; ils sont trop faibles pour départager les configurations.

**Mesure — comparaison aux chiffres publiés.** Notre meilleur score de développement est 58.80. Le chiffre publié pour BEATs gelé est 60.28, pour NA-BEATs 63.18, pour le meilleur système MERL 66.20. Notre front-end gelé se situe 1.48 point sous le BEATs gelé publié. Comparaison croisée, non une mesure.

---

## 6. Résultats négatifs

Quatre hypothèses ont été testées et rejetées. Elles occupent cette section entière parce que trois d'entre elles portaient sur l'exploitation du second canal, c'est-à-dire sur la nouveauté même du challenge 2026.

### 6.1 Hypothèse 1 — Masque inter-canaux : rejetée, effet neutre

**Raisonnement initial.** Le rapport de puissance entre canaux vaut `1 + SNR`. Le gain de Wiener correspondant s'écrit `G = 1 − P₂/P₁`. Pondérer le mel par ce gain devait atténuer le bruit et préserver la machine, l'anomalie étant émise depuis la même position physique que la machine, donc spatialement cohérente avec elle.

**Première implémentation inerte.** L'analyse des rapports d'énergie par machine l'a révélé, colonne `q1_raies_dB` de `e0_results/e0b_summary.csv`. Quand `P₂ > P₁` le gain sature au plancher sur tout le spectre ; quand `P₁ ≫ P₂` il vaut quasiment 1. Dans les deux cas le masque ne modifie rien après normalisation de l'entrée du modèle.

**Mesure.** Le rapport Q1 sur les raies machine est **négatif pour quatre machines sur sept** : valveEmu −3.60 dB, fan −3.19 dB, gearboxEmu −2.83 dB, sliderEmu −0.27 dB. Le canal 2 y est donc plus fort que le canal 1 sur les raies attribuées à la machine. Seules ToyCar (+19.53 dB) et ToyCarEmu (+11.22 dB) présentent l'écart attendu.

**Correction apportée.** Calibration de l'écart de gain entre microphones, estimée par clip sur un percentile bas du rapport `P₁/P₂`, c'est-à-dire dans les bandes où la machine est absente. Fonction `wiener_gain`, section 4.7.

**Mesure après correction.** À configuration strictement appariée — même pooling `freq`, même agrégation `max`, même `--per-domain` :

| Jeu | Sans masque | Avec masque `leveldiff` | Δ |
|---|---|---|---|
| Développement | 57.91 | 58.35 | **+0.44** |
| Évaluation | 56.75 | 56.67 | **−0.08** |

**Conclusion : le masque est neutre.** Le signe s'inverse entre les deux jeux et l'amplitude reste sous le demi-point dans les deux cas. Ni effet nul, ni effet favorable. Une conclusion favorable tirée du seul développement serait démentie par l'évaluation.

**Mesure complémentaire.** Sous pooling `mean`, sans `--per-domain`, le masque est effectivement sans effet mesurable :

| Configuration | Score |
|---|---|
| Sans masque | 56.54 |
| `leveldiff`, α = 1.0 | 56.55 |
| `leveldiff`, α = 0.5 | 56.81 |

L'écart à α = 1 vaut +0.01 point, soit l'arrondi.

**Mesure — contrôle par inversion des canaux.** `--swap-channels` traite le canal 2 comme le micro proche.

| Configuration | Score |
|---|---|
| `leveldiff`, `per-domain` | 58.35 |
| `leveldiff`, `per-domain`, `--swap-channels` | 50.76 |
| Δ | **−7.59** |

**Interprétation.** L'inversion dégrade fortement. Le canal 1 est donc bien le canal informatif, malgré les rapports d'énergie négatifs mesurés sur quatre machines. Ces deux constats coexistent et doivent être rapportés ensemble : le canal 2 peut porter plus d'énergie sur les raies machine tout en étant moins informatif pour la détection.

**Hypothèse explicative du résultat neutre.** Altérer le spectrogramme détruit autant d'information qu'elle en nettoie. Le modèle pré-entraîné a appris des textures spectro-temporelles que la repondération par bande dégrade. Le bénéfice du débruitage et le coût de la distorsion se compensent.

### 6.2 Hypothèse 2 — FarMix : rejetée, retournement entre les jeux

**Raisonnement initial.** Le canal 2 constitue une banque de bruit réelle, en domaine, enregistrée dans la même salle avec le même matériel, et gratuite. Plutôt que de nettoyer l'entrée, on enrichit la banque de référence avec des variantes bruitées des clips normaux.

**Mesure — développement.** À `per-domain`, pooling `freq`, agrégation `max`.

| Configuration | Score | Δ vs `per-domain` |
|---|---|---|
| Sans FarMix | 57.91 | référence |
| FarMix 2, SNR −5–5 dB | 57.96 | +0.05 |
| FarMix 2, SNR 10–30 dB | 57.55 | −0.36 |
| FarMix 1, SNR 0–20 dB | 58.36 | +0.45 |
| FarMix 2, SNR 0–20 dB | 58.33 | +0.42 |
| FarMix 4, SNR 0–20 dB | 58.69 | +0.78 |
| FarMix 2, SNR 0–10 dB | **58.80** | **+0.89** |

Le gain est faible et présente un optimum net sur la plage de SNR : 0–10 dB donne +0.89, alors que 10–30 dB donne −0.36 et −5–5 dB donne +0.05.

**Mesure — évaluation.**

| Soumission | Score officiel | AUC source | AUC cible |
|---|---|---|---|
| `perdom` | 56.75 | 59.98 | 56.00 |
| `perdom_fm2` (SNR 0–10 dB) | **54.40** | 60.98 | 49.43 |
| Δ | **−2.35** | +1.00 | **−6.57** |

**Conclusion : retournement de signe.** FarMix gagne 0.89 point sur le développement et en perd 2.35 sur l'évaluation, avec la même plage de SNR. C'est le seul retournement franc mesuré dans ce travail. La perte se localise entièrement sur l'AUC cible, qui chute de 6.57 points.

**Hypothèse explicative.** FarMix multiplie chaque banque par `N+1`. La banque cible ne compte que 10 clips de départ. Ses variantes ne sont donc que `N+1` versions du même très petit ensemble : sa densité apparente augmente sans que sa couverture réelle progresse. La normalisation par domaine estime son échelle sur ces distances artificiellement resserrées, et l'échelle de la banque cible devient fausse. La localisation de la perte sur l'AUC cible est cohérente avec cette hypothèse, sans la démontrer.

### 6.3 Hypothèse 3 — CNN entraîné de zéro : rejetée

**Raisonnement initial.** Un CNN discriminatif entraîné sur la classification machine × attributs, avec perte ArcFace, devait produire des embeddings mieux adaptés au domaine que ceux d'un modèle pré-entraîné sur AudioSet, pour 70 fois moins de paramètres.

**Mesure.** Les trois runs valides, tous en mode déterministe. Ce sont les seuls chiffres CNN de ce rapport.

| Configuration | Graine | Classes | Paramètres | Score officiel |
|---|---|---|---|---|
| `--pseudo-labels 16` | 0 | 98 | 1 218 080 | **52.29** |
| `--pseudo-labels 16` | 1 | 98 | 1 218 080 | **49.04** |
| `--pseudo-labels 0` | 0 | 38 | 1 210 400 | **50.21** |

Les trois valeurs ont été recalculées indépendamment depuis les `scores_*.npz` courants et coïncident au centième avec les journaux, les résumés JSON et `results_dev.csv`.

Les trois configurations restent **5.62 à 8.87 points** sous la configuration AST retenue (57.91).

**Mesure — reproductibilité à graine fixée.** Deux exécutions par graine pour `--pseudo-labels 16`, journaux `run_cnn_pl16_s0_r1/r2` et `run_cnn_pl16_s1_r1/r2`. Métriques par machine, score officiel et trajectoire d'apprentissage **identiques** entre les deux exécutions, pour les deux graines. Le mode déterministe fonctionne.

**L'effet des pseudo-étiquettes n'est pas établi.**

| Comparaison | Valeur |
|---|---|
| `--pseudo-labels 16`, graine 0 | 52.29 |
| `--pseudo-labels 0`, graine 0 | 50.21 |
| Écart apparent | **+2.08** |
| Sensibilité à la graine de la branche | **3.25** |

L'écart apparent de 2.08 points est **inférieur à la sensibilité à la graine de 3.25 points**, et chaque condition n'a été mesurée qu'à une seule graine. L'écart n'est donc pas interprétable. Il faut l'écrire explicitement : ces mesures ne permettent pas de conclure que les pseudo-étiquettes aident, ni qu'elles nuisent. Trancher demande cinq graines par condition au minimum.

**Mesure — deux trajectoires d'apprentissage opposées.** C'est le fait le plus instructif de cette branche.

| Époque | `pl16` perte | `pl16` accuracy | `pl0` perte | `pl0` accuracy |
|---|---|---|---|---|
| 1 | 12.1973 | 0.296 | 9.4371 | 0.283 |
| 8 | 0.7757 | **1.000** | 2.6653 | 0.157 |
| 16 | 0.7757 | 1.000 | 2.6078 | 0.237 |
| 24 | 0.7757 | 1.000 | 2.5122 | 0.237 |
| 32 | 0.7756 | 1.000 | 2.4715 | 0.333 |
| 40 | 0.7756 | 1.000 | 2.4573 | **0.427** |

Les deux configurations échouent, mais **par des mécanismes opposés**.

Avec 98 classes, le réseau atteint une accuracy de **1.000 dès l'époque 8**. La perte se fige à 0.7757 et ne bouge plus que de 0.0001 sur les 32 époques suivantes. L'ensemble d'entraînement est mémorisé au cinquième du budget, et le reste de l'entraînement n'apporte rien.

Avec 38 classes, le réseau **n'apprend presque pas**. L'accuracy reste à 0.427 à l'époque 40, après un minimum à 0.157 à l'époque 8. La perte décroît encore à l'époque 40 (2.4573) : l'entraînement n'a pas convergé.

**Conséquence pour l'hypothèse de mémorisation.** La configuration qui mémorise complètement obtient **52.29**, la configuration qui ne mémorise pas du tout obtient **50.21**. La mémorisation n'explique donc pas l'ordre des scores, et n'est pas la cause commune de l'échec de la branche. Elle décrit un des deux régimes défaillants, pas les deux.

**Hypothèse — deux régimes défaillants distincts.** À 98 classes, les pseudo-classes sont petites et déséquilibrées, jusqu'à 2 ou 3 clips ; la discrimination devient triviale et la marge angulaire d'ArcFace projette tous les clips d'une classe sur un même point de la sphère. Les clips normaux et anormaux d'une même machine y atterrissent ensemble et la distance au plus proche voisin ne mesure plus rien. Le plateau de perte à 0.7757, très au-dessus de zéro, est cohérent avec une marge saturée. À 38 classes, quatre machines forment chacune une classe unique de 1000 clips ; la tâche demande de séparer des classes très hétérogènes et le réseau, à ce budget, n'y parvient pas. Aucun des deux régimes ne produit d'embedding utile à la détection, pour des raisons opposées. Un budget d'époques, une taille de modèle ou un nombre de pseudo-classes intermédiaires pourraient exister entre les deux ; ce travail ne les a pas cherchés.

**Mesure — la machine `fan` sous CNN.** AUC source 50.24 (`pl16` graine 0), 43.84 (`pl16` graine 1), 43.16 (`pl0`) ; AUC cible 50.70, 53.38 et 55.18. Le CNN n'obtient rien d'exploitable de plus que la branche AST sur cette machine, et l'écart entre graines dépasse l'écart au hasard.

### 6.4 `--domain-align` : rejetée, dégradation constante

**Mesure.** À pooling `freq`, agrégation `max`.

| Jeu | Sans correction | `--domain-align` | Δ |
|---|---|---|---|
| Développement | 58.56 | 58.29 | **−0.27** |
| Évaluation | 52.28 | 49.96 | **−2.32** |

Ce n'est pas un retournement mais une **dégradation constante**, faible sur le développement et franche sur l'évaluation. Le recentrage des références cible sur la moyenne source dégrade dans les deux jeux.

**Mesure.** L'effet le plus net porte sur l'AUC cible de ToyDrone, qui passe de 27.80 sans correction à **21.48** avec `--domain-align`, soit 28.5 points sous le hasard.

**Hypothèse.** Le recentrage déplace les références cible sans réduire la dispersion de la banque source. Le déséquilibre 990 contre 10 subsiste, et l'appariement continue de se faire majoritairement côté source, sur des références désormais mal placées.

### 6.5 Élargissement du voisinage : rejeté

**Mesure.** À `per-domain`, pooling `freq`, agrégation `max`.

| `--knn` | Score officiel | hm AUC source | hm AUC cible | hm pAUC |
|---|---|---|---|---|
| 1 | **57.91** | 61.52 | 61.00 | 52.20 |
| 2 | 57.15 | 60.58 | 59.36 | 52.25 |
| 8 | 55.60 | 58.63 | 56.71 | 51.91 |

Dégradation monotone, −0.76 point de `k = 1` à `k = 2`, −2.31 points de `k = 1` à `k = 8`. La dégradation touche l'AUC source et l'AUC cible dans les mêmes proportions ; la pAUC est presque insensible.

**Hypothèse.** La banque cible ne compte que 10 références. Avec `k = 8`, la distance cible moyenne huit voisins sur dix, soit presque toute la banque : l'estimateur devient une distance au centroïde et perd sa sensibilité locale. La dégradation symétrique sur le domaine source suggère toutefois un effet plus général de lissage, non spécifique au déséquilibre.

### 6.6 La machine `fan` reste au niveau du hasard

**Mesure.** Plages observées sur les 29 configurations AST du développement.

| Machine | AUC source | AUC cible | pAUC |
|---|---|---|---|
| `fan` | **47.40 – 55.12** | **43.36 – 50.56** | **48.58 – 51.89** |
| ToyCar | 54.32 – 68.92 | 45.60 – 73.92 | 49.58 – 58.53 |
| ToyCarEmu | 48.44 – 60.48 | 46.80 – 79.12 | 48.84 – 56.32 |
| bearingEmu | 53.16 – 64.80 | 49.16 – 62.12 | 52.74 – 61.21 |
| gearboxEmu | 52.08 – 68.44 | 53.16 – 65.64 | 49.53 – 54.84 |
| sliderEmu | 47.00 – 66.04 | 43.44 – 60.68 | 49.68 – 55.53 |
| valveEmu | 63.04 – 82.60 | 52.40 – 71.20 | 51.68 – 57.37 |

`fan` est la seule machine dont les trois métriques restent bornées près de 50 dans toutes les configurations AST. Son AUC cible n'y dépasse jamais 50.56, c'est-à-dire jamais le hasard. Son AUC source plafonne à 55.12, atteinte par le pooling `meanstd`, une configuration par ailleurs médiocre. Dans la configuration retenue, `fan` donne 50.20 / 47.24 / 48.68.

**Nuance mesurée.** Les deux runs CNN valides donnent une AUC cible de 50.70 et 53.38 sur `fan`, donc au-dessus de 50. Mais leurs AUC source valent 50.24 et 43.84, et l'écart entre graines de la branche CNN est de 3.25 points. Aucune configuration ne produit donc une détection au-dessus du hasard **de façon cohérente entre les deux domaines**, et le 53.38 n'est pas séparable du bruit de graine. La formulation exacte est donc : aucune configuration testée n'obtient de détection exploitable sur `fan`, et non que toute valeur mesurée reste sous 50.

Aucun levier testé ne modifie ce constat : ni le masque, ni le pooling, ni le voisinage, ni FarMix, ni la normalisation par domaine, ni le changement de front-end.

**Chiffre publié, pour situation.** L'ablation MERL rapporte 52.12 pour BEATs gelé et 53.15 pour NA-BEATs sur `fan`. Le système gagnant n'obtient donc pas davantage sur cette machine. Comparaison croisée, non une mesure.

**Hypothèse.** Le ventilateur est une source stationnaire large bande, spectralement proche du bruit d'usine. Ni la représentation ni le scoring ne disposent d'un contraste exploitable. Cette hypothèse est la même que celle de la section 2.2. `fan` présente la deuxième modulation temporelle la plus faible des sept machines, 2.62 dB, derrière gearboxEmu à 1.94 dB. La correspondance n'est donc pas stricte : gearboxEmu, plus stationnaire encore au sens de ce critère, atteint 68.44 d'AUC source dans sa meilleure configuration. La modulation temporelle seule n'explique pas l'échec sur `fan`.

---

## 7. Observations méthodologiques

### 7.1 Reproductibilité et sensibilité à la graine : deux propriétés distinctes

Il faut séparer deux questions souvent confondues. Une même exécution répétée donne-t-elle le même résultat ? Et un changement de graine change-t-il le résultat ? Les deux branches se comportent différemment sur la seconde, et identiquement sur la première depuis le correctif de déterminisme.

**Mesure — reproductibilité à graine fixée.** Les deux branches sont reproductibles.

| Branche | Répétitions mesurées | Résultat |
|---|---|---|
| AST gelé | 3 graines, configuration retenue | 57.91 aux trois, au centième |
| CNN, graine 0 | 2 exécutions | 52.29 aux deux ; métriques par machine et trajectoire identiques |
| CNN, graine 1 | 2 exécutions | 49.04 aux deux ; métriques par machine et trajectoire identiques |

**Fait rapporté, non vérifiable dans le dépôt.** Avant le forçage du mode déterministe, les convolutions cuDNN étaient non déterministes par défaut et deux exécutions du CNN à graine fixée différaient jusqu'à 2.66 points. Cette mesure a été relevée par l'équipe ; les exécutions concernées n'ont pas été conservées et le chiffre n'est pas reproductible depuis les fichiers actuels. Il est cité ici pour justifier le correctif, non comme une mesure de ce rapport.

**Mesure — sensibilité à la graine.** C'est là que les deux branches divergent.

| Branche | Écart entre graines |
|---|---|
| AST gelé | **0.00** (57.91 sur les graines 0, 1, 2) |
| CNN | **3.25** (52.29 à la graine 0, 49.04 à la graine 1) |

La branche AST est invariante par construction : les poids sont gelés, le scoring est un calcul de plus proches voisins, la graine ne pilote rien. La branche CNN dépend de son initialisation, de l'ordre des batches, du tirage des crops, de SpecAugment et du k-means des pseudo-étiquettes.

**Conséquence pour la lecture de tout ce rapport.** Les écarts entre configurations AST sont réels dès qu'ils dépassent l'arrondi, même faibles : +0.47 point pour l'agrégation par maximum, +0.44 pour le masque sur le développement, −0.08 sur l'évaluation. Aucun de ces écarts n'est du bruit de graine, puisque ce bruit est nul. En revanche, aucun écart inférieur à 3.25 points n'est interprétable entre deux configurations CNN. Cette asymétrie doit être énoncée avant toute discussion des écarts, faute de quoi les deux branches seraient lues avec le même barème.

Le correctif de déterminisme ne réduit pas cette asymétrie ; il la rend mesurable. Sans lui, les 3.25 points d'écart entre graines étaient mêlés à jusqu'à 2.66 points de variabilité d'exécution, et les deux sources étaient indissociables.

Elle ne rend pas les petits écarts AST *significatifs* pour autant. Ils sont exacts et reproductibles, mais mesurés sur un seul jeu de 1400 clips de test ; leur généralisation à d'autres machines n'est pas établie, et la section 5.4 montre qu'ils ne suffisent pas à sélectionner un système.

### 7.2 Transfert des hyperparamètres du développement vers l'évaluation

C'est une contribution méthodologique à part entière. Deux réglages se comportent différemment selon le jeu, et les deux cas sont de nature différente.

| Réglage | Développement | Évaluation | Nature |
|---|---|---|---|
| `--domain-align` | −0.27 | −2.32 | dégradation constante, amplifiée |
| `--per-domain` | −0.65 | +4.47 | **retournement favorable** |
| FarMix 2, SNR 0–10 dB | +0.89 | −2.35 | **retournement défavorable** |
| Masque `leveldiff` | +0.44 | −0.08 | neutre, signe non stable |

Trois régimes distincts apparaissent. `--domain-align` dégrade dans les deux jeux, l'amplitude seule change : sélectionner sur le développement conduit à la bonne décision, l'écarter. FarMix se retourne : sélectionner sur le développement conduit à l'adopter, et coûte 2.35 points. `--per-domain` se retourne dans l'autre sens : sélectionner sur le développement conduit à l'écarter, et coûte 4.47 points.

**Conséquence.** Sur ce jeu, le développement n'est pas un prédicteur fiable du signe d'un effet, encore moins de son amplitude. Un écart de moins d'un point sur le développement ne permet aucune décision.

**Mesure — les baselines officielles présentent la même instabilité.** Les chiffres publiés donnent MSE 56.66 sur le développement et 59.80 sur l'évaluation, Mahalanobis 57.66 puis 54.76. Le classement s'inverse entre les deux jeux, avec un écart de 2.90 points sur l'évaluation. L'instabilité n'est donc pas un artefact de notre système. Comparaison de chiffres publiés entre eux, non une mesure.

**Hypothèse.** La contrainte *first-shot* est en cause. Les cinq machines de l'évaluation sont absentes du développement. Tout réglage ajusté sur sept machines connues encode des propriétés de ces sept machines, pas des propriétés du problème. Les réglages qui survivent au transfert sont ceux qui corrigent un défaut structurel du protocole — le déséquilibre 990 contre 10 — plutôt qu'une caractéristique des données.

Cette hypothèse est cohérente avec l'ordre observé : `--per-domain`, qui traite le déséquilibre, gagne au transfert ; FarMix, qui exploite la statistique du bruit des machines vues, perd.

### 7.3 Le F1 récompense le comportement dégénéré

**Mesure.** Le jeu de test contient exactement 50 % d'anomalies, sur les deux jeux et pour toutes les machines (section 3.1). Tout déclarer anormal donne donc mécaniquement :

| Métrique | Valeur du système dégénéré |
|---|---|
| Précision | 50.00 |
| Rappel | 100.00 |
| **F1** | **66.67** |

**Mesure.** Aucune de nos sept soumissions n'atteint ce F1. La meilleure, `thr_gamma07`, obtient 51.78 en moyenne harmonique de F1. Le seuillage par défaut donne 34.89.

| Soumission | hm F1 | hm précision | hm rappel | Score officiel |
|---|---|---|---|---|
| `thr_gamma07` | **51.78** | 55.95 | 48.18 | 56.75 |
| `perdom_fm2` | 47.89 | 53.72 | 43.20 | 54.40 |
| `thr_otsu` | 41.65 | 55.99 | 33.16 | 56.75 |
| `leveldiff` | 36.47 | 64.84 | 25.37 | 56.67 |
| `perdom` | 34.89 | 63.20 | 24.10 | 56.75 |
| `align` | 34.78 | 60.08 | 24.48 | 49.96 |
| `noalign` | 34.55 | 60.59 | 24.17 | 52.28 |

Un système qui discrimine honnêtement est donc **classé en dessous** d'un système qui ne discrimine pas du tout, sur cette métrique. Le F1 n'ordonne pas non plus les soumissions comme le score officiel : `perdom` et `thr_gamma07` ont le même score officiel de 56.75 et des F1 de 34.89 et 51.78 ; `align`, la plus faible en score officiel, a un F1 supérieur à celui de `perdom`.

**Mesure — la démonstration la plus nette.** Soumission `align`, machine ToyDrone, domaine cible.

| Grandeur | Valeur |
|---|---|
| AUC | **21.48** |
| Précision | 50.52 |
| Rappel | **98.00** |
| **F1** | **66.67** |

Le F1 atteint exactement la valeur du système dégénéré, avec un rappel de 98 %, sur une case dont l'AUC vaut 21.48. Une AUC de 21.48 signifie que l'ordonnancement des scores est **inversé** : les clips normaux reçoivent des scores plus élevés que les clips anormaux. Le système obtient donc un F1 maximal sur une détection inversée. Le mécanisme est direct : le seuil est franchi par 98 % des clips anormaux, mais aussi par presque tous les normaux, ce que le F1 ne pénalise pas quand la classe positive représente la moitié du jeu.

**Mesure — le taux de déclaration confirme le mécanisme.** Sur ToyDrone, `align` déclare 52 % des 200 clips anormaux. La proportion attendue d'un classifieur informatif serait proche de 50 % avec une précision nettement supérieure à 50.52.

**Mesure — le balayage du quantile est monotone.** Sur le développement, à `per-domain`, pooling `freq`, agrégation `max`.

| Stratégie | hm F1 | hm précision | hm rappel |
|---|---|---|---|
| `gamma` q = 0.40 | **61.56** | 56.68 | 67.36 |
| `gamma` q = 0.50 | 56.06 | 57.28 | 54.90 |
| `gamma` q = 0.60 | 51.35 | 58.83 | 45.56 |
| `gamma` q = 0.70 | 42.75 | 59.01 | 33.51 |
| `gamma` q = 0.80 | 28.35 | 56.46 | 18.92 |
| `gamma` q = 0.90 | 23.89 | 63.85 | 14.69 |
| `gamma` q = 0.95 | 9.72 | 54.69 | 5.34 |
| `otsu` | 48.36 | 61.27 | 39.95 |
| `mixture` | 39.81 | 61.23 | 29.49 |

Le F1 décroît strictement de q = 0.40 à q = 0.95, de 61.56 à 9.72. **Aucun optimum intérieur.** Le rappel décroît de 67.36 à 5.34 ; la précision reste comprise entre 54.69 et 63.85 sans tendance nette.

**Interprétation.** L'absence d'optimum indique que l'optimisation du quantile converge vers la dégénérescence et non vers un bon seuil. Baisser q augmente le F1 en augmentant le rappel, et la limite de ce mouvement est le système qui déclare tout anormal, à 66.67. Le meilleur point mesuré, 61.56, reste 5.11 points sous cette limite : le balayage s'arrête avant la dégénérescence complète, mais il pointe dans sa direction.

**Le score officiel est insensible à tout ceci.** Les neuf stratégies de la table ci-dessus donnent le même score officiel de 57.91, puisque ni l'AUC ni la pAUC ne dépendent du seuil. Le seuillage est donc un problème réel de déploiement, sans effet sur la métrique de classement.

**Chiffre externe, à citer comme tel.** Plusieurs équipes du classement officiel affichent exactement 66.67 de F1 avec un rappel de 100 %. Cette information ne provient pas du dépôt et n'a pas été vérifiée par nous.

**Formulation retenue.** Il s'agit d'une observation sur le protocole, étayée par les mesures ci-dessus, et non d'une critique des équipes concernées. La métrique F1, appliquée à un jeu équilibré à 50 %, attribue sa valeur maximale accessible sans discrimination à un comportement qui n'en comporte aucune. Le score officiel, fondé sur AUC et pAUC, n'a pas ce défaut.

### 7.4 Un défaut d'implémentation du seuillage, et sa trajectoire complète

La section serait incompréhensible sans le récit complet ; il comporte trois étapes.

**Étape 1 — le score de référence valait la moitié de sa valeur.** Le score des clips normaux d'entraînement, sur lequel le seuil est calibré, moyennait les `k+1` plus proches voisins sans écarter la distance du clip à lui-même, nulle par construction. Avec `k = 1`, la moyenne portait donc sur `{0, d₁}` au lieu de `{d₁}`. Le score de référence valait la moitié de sa vraie valeur, et le seuil calibré dessus était deux fois trop bas.

**Étape 2 — après correction, le seuil par défaut devient trop conservateur.** Le paramètre `skip_self` écarte désormais cette distance nulle. Le score de référence double, le seuil monte. À q = 0.90, la valeur par défaut, le rappel tombe à 14.69 et le F1 à 23.89 sur le développement. Le seuil est passé de deux fois trop bas à trop haut.

**Étape 3 — le réglage du quantile corrige, mais sans optimum.** Descendre q rétablit le rappel : q = 0.40 donne 67.36 de rappel et 61.56 de F1. Mais le balayage est monotone, sans optimum intérieur (table de la section 7.3). Le réglage compense le conservatisme sans identifier un bon seuil.

**Étape 4 — l'ajustement de la loi gamma échouait en configuration `--per-domain`.** Défaut découvert lors de l'audit. `scipy.stats.gamma.fit` était appelé avec `floc = min(scores) − 1e-9`. Avec `--per-domain`, les scores sont normalisés par banque, donc resserrés autour de 1 et strictement positifs : cette marge de `1e-9` ne laissait aucun degré de liberté et l'ajustement échouait. Le code se rabattait alors silencieusement sur le quantile empirique. Le journal de la campagne précédente enregistre **146 replis**, concentrés sur les configurations `--per-domain` : 129 sur 154 couples (run, machine) côté développement, 15 sur 15 côté évaluation, contre 2 sur 56 pour les configurations sans `--per-domain`. La stratégie annoncée comme « loi gamma » était donc, en pratique et pour toute configuration `--per-domain`, un quantile empirique.

**Correction apportée.** `choose_threshold` essaie désormais `floc = 0` avant `floc = min(scores) − 1e-6`. L'ajustement converge sur les scores normalisés par banque.

**Mesure de l'effet de la correction.** Les colonnes de F1 des sept configurations `gamma` de `results_dev.csv` ont toutes changé après correction, tandis que les configurations `otsu` et `mixture` sont inchangées au centième — 48.36 et 39.81 avant comme après. C'est la signature attendue : ces deux stratégies n'appellent pas l'ajustement gamma. Le F1 à q = 0.90 passe de 25.97 à 23.89, celui à q = 0.95 de 18.51 à 9.72.

| Stratégie | hm F1 avant correction | hm F1 après correction |
|---|---|---|
| `gamma` q = 0.40 | 62.36 | 61.56 |
| `gamma` q = 0.50 | 59.90 | 56.06 |
| `gamma` q = 0.60 | 55.49 | 51.35 |
| `gamma` q = 0.70 | 50.84 | 42.75 |
| `gamma` q = 0.80 | 41.63 | 28.35 |
| `gamma` q = 0.90 | 25.97 | 23.89 |
| `gamma` q = 0.95 | 18.51 | 9.72 |
| `otsu` | 48.36 | 48.36 |
| `mixture` | 39.81 | 39.81 |

La soumission `thr_gamma07` est donc bien un quantile de loi gamma, et non un 70ᵉ percentile empirique. Le score officiel est inchangé par cette correction, le seuil n'entrant ni dans l'AUC ni dans la pAUC.

**Étape 5 — l'ajustement échoue encore sur la branche CNN.** La correction règle le cas AST mais pas le cas CNN, dont la distribution des scores diffère.

| Run | Machines en repli sur le percentile | Machines à F1 nul |
|---|---|---|
| CNN `pl16`, graine 0 | **3 sur 7** : ToyCar, ToyCarEmu, fan | 0 |
| CNN `pl16`, graine 1 | **0 sur 7** | **5** |
| CNN `pl0`, graine 0 | **2 sur 7** : ToyCarEmu, gearboxEmu | **3** |
| AST, toutes configurations courantes | 0 | 0 |

**Observation contre-intuitive, et c'est le point important.** Le repli sur le percentile n'est pas la cause de l'échec de seuillage. La corrélation va dans l'autre sens. Le run avec le plus de replis, `pl16` graine 0 avec 3 sur 7, est le seul des trois sans aucune machine à F1 nul. Le run où l'ajustement gamma **converge sur les sept machines**, `pl16` graine 1, est celui qui compte cinq machines à F1 nul. Un ajustement qui converge peut donc produire un seuil pire qu'un repli sur le percentile empirique. La convergence de l'estimateur ne garantit rien sur la qualité du seuil qui en découle.

### 7.4 bis La dégénérescence inverse : seuil au-dessus de tous les scores

La section 7.3 décrit une métrique qui récompense le système déclarant tout anormal. Le symptôme inverse existe aussi et est mesuré ici.

**Mesure.** Branche CNN, graine 1, en mode déterministe.

| Machine | AUC source | AUC cible | Précision | Rappel | F1 |
|---|---|---|---|---|---|
| ToyCar | 33.44 | 53.30 | 0.00 | **0.00** | **0.00** |
| ToyCarEmu | 48.92 | 58.90 | 0.00 | **0.00** | **0.00** |
| bearingEmu | 63.32 | 58.38 | 81.48 | 22.00 | 34.65 |
| fan | 43.84 | 53.38 | 0.00 | **0.00** | **0.00** |
| gearboxEmu | 36.38 | 55.98 | 0.00 | **0.00** | **0.00** |
| sliderEmu | 44.08 | 42.96 | 16.67 | 1.00 | 1.89 |
| valveEmu | 56.96 | 50.46 | 0.00 | **0.00** | **0.00** |

**Cinq machines sur sept** affichent un F1 de 0.00 avec un rappel nul. Le seuil est passé au-dessus de tous les scores de test : aucun clip n'est déclaré anormal. Le système ne détecte rien du tout.

**Le symptôme n'est pas isolé.** La configuration `--pseudo-labels 0` le présente aussi, sur trois machines : ToyCar, ToyCarEmu et gearboxEmu, toutes trois à précision, rappel et F1 nuls. Deux des trois runs CNN valides sont donc touchés.

| Run CNN | hm F1 | hm rappel | Machines à rappel nul |
|---|---|---|---|
| `pl16`, graine 0 | 3.97 | 2.11 | 0 sur 7 |
| `pl16`, graine 1 | 3.58 | 1.91 | **5 sur 7** |
| `pl0`, graine 0 | 12.20 | 6.99 | **3 sur 7** |

Le run le plus touché en nombre de machines n'est pas celui qui a le F1 le plus bas : `pl0` compte trois machines à zéro et obtient pourtant le meilleur F1 harmonique des trois, 12.20. Les quatre machines restantes y sont mieux seuillées.

**Interprétation.** Les deux dégénérescences encadrent le problème du seuillage non supervisé. À un extrême, tout est déclaré anormal : le rappel vaut 100, la précision tombe à la proportion d'anomalies, et le F1 atteint 66.67 sur un jeu équilibré. À l'autre, rien n'est déclaré anormal : rappel, précision et F1 valent zéro. Entre les deux, aucune des quatre stratégies testées ne trouve un point stable, et le balayage du quantile est monotone (section 7.3). Le problème n'est pas le choix de la stratégie mais l'absence de tout signal permettant de le résoudre sans étiquette.

**Chiffre externe, à citer comme tel.** Certaines équipes du classement officiel affichent un F1 nul. Cette information ne provient pas du dépôt et n'a pas été vérifiée par nous. Elle indique que les deux dégénérescences se rencontrent dans les soumissions réelles, aux deux bouts du classement en F1.

**Ce que cet épisode illustre.** Un repli silencieux sur une méthode de secours transforme la nature d'un résultat sans le signaler. Le message de repli existait dans le code et était capturé par le journal ; il n'avait pas été relu. La leçon opérationnelle est de traiter tout repli comme une erreur à faire remonter, et non comme un avertissement.

### 7.5 Sélection du système retenu

**Mesure.** La configuration au meilleur score sur le développement n'est pas le système retenu.

| | Développement | Évaluation |
|---|---|---|
| Meilleure config dev : `per-domain` + FarMix 2, SNR 0–10 dB | **58.80** | 54.40 |
| Système retenu : `per-domain` sans FarMix | 57.91 | **56.75** |
| Δ | +0.89 en faveur de la première | +2.35 en faveur du second |

Le système retenu est `per-domain` sans FarMix. Il est 0.89 point derrière sur le développement et 2.35 points devant sur l'évaluation.

C'est une illustration directe du problème de la section 7.2. Une sélection faite sur le développement seul aurait retenu FarMix et perdu 2.35 points. La décision a été prise sur l'évaluation, ce qui n'est possible que parce que le challenge est clos et que l'évaluateur officiel est disponible. Dans une campagne réelle, cette information n'existerait pas au moment du choix.

**Conséquence pour la publication.** Le protocole de sélection doit être déclaré explicitement. Annoncer 56.75 comme le résultat d'un système sélectionné sur le développement serait inexact.

### 7.6 Traçabilité de la campagne

**Mesure — état courant, vérifié.** Les trois lignes CNN de `results_dev.csv` sont à jour et valides. Le fichier est daté du 2026-09-03 13:58, postérieur aux trois runs CNN (11:38, 11:42, 13:57). Les trois résumés JSON portent `deterministic: true`. Les 32 lignes du CSV coïncident au centième avec le recalcul indépendant depuis les scores bruts.

Un audit intermédiaire avait relevé trois lignes CNN périmées ou invalides, corrigées depuis : deux par réagrégation, la troisième — `--pseudo-labels 0` — par relance complète du run, ses fichiers étant antérieurs au correctif de l'ajustement gamma comme au forçage du déterminisme. Cet épisode motive la recommandation de la section 9.4 : comparer systématiquement l'horodatage des fichiers agrégés à celui des résultats qu'ils agrègent, et vérifier la présence du champ `deterministic` dans chaque résumé.

**Mesure.** Trois écarts de traçabilité subsistent et sont déclarés ici.

Premièrement, `run_all.log` est antérieur aux résultats qu'il est censé documenter. Il est daté du 2026-09-02 18:17 ; `asd_pipeline.py`, `collect_results.py` et `run_all.sh` sont du 2026-09-03 10:16, les fichiers de résultats du 2026-09-03 10:42 à 10:45, les CSV agrégés du 10:54. Le journal décrit donc la campagne précédente. Il ne contient ni la soumission `leveldiff`, ni les runs de graines, ni les lignes d'époque du CNN que le filtre de `run_all.sh` laisse désormais passer. Les résultats eux-mêmes sont complets et vérifiés ; c'est leur journal qui manque.

Il ne contient pas non plus les cinq exécutions CNN valides, dont les journaux sont conservés séparément sous `run_cnn_pl16_s{0,1}_r{1,2}.log` et `run_cnn_pl0_s0.log`. Ces cinq journaux sont complets : ils contiennent la trajectoire d'apprentissage, les tailles de pseudo-classes, les messages de repli de l'ajustement gamma et les métriques par machine. Toutes les mesures CNN de ce rapport en proviennent.

Deuxièmement, `run_all.sh` filtre la sortie du pipeline par `grep`. Ce filtre est nécessaire pour la lisibilité, mais il élimine des informations utiles à la vérification : formes des tenseurs, nombre de pseudo-classes et leurs tailles, grille de tokens. Sa liste de motifs a été élargie mais la campagne n'a pas été rejouée avec le journal conservé.

Troisièmement, une collision de tag bénigne existe dans la campagne. Le run `--per-domain` de la section « scoring » et le run `--threshold-q 0.9` du balayage produisent le même tag, 0.9 étant la valeur par défaut. Le second écrase le premier. Les deux configurations étant identiques, le résultat est identique et aucun chiffre n'est faussé ; la campagne compte simplement une configuration distincte de moins que d'invocations.

**Mesure — ce qui est en revanche entièrement vérifié.** Les 32 triplets du développement et les 7 de l'évaluation sont complets et sans doublon de tag. Le score officiel des 32 configurations du développement a été recalculé indépendamment depuis les scores bruts, avec un écart maximal de 0.0047 point. La cohérence entre `official_score`, `hm_AUC` et `hm_pAUC` est vérifiée sur les 7 soumissions. `results_eval.csv` ne contient que la génération courante ; aucune trace des six soumissions de la génération précédente, conservées séparément, n'y figure.

---

## 8. Limites

Les limites sont classées par ordre de gravité pour la portée des conclusions.

**8.1 Le système reste sous la baseline autoencodeur sur l'évaluation.** Notre meilleur score est 56.75, contre 59.80 pour la baseline MSE selon les chiffres publiés, soit 3.05 points de déficit. La contribution de ce travail est méthodologique — la normalisation par domaine et l'analyse du transfert — et non un gain de performance absolue.

**8.2 Les écarts entre les meilleures configurations du développement ne permettent pas de sélectionner un système.** Les six premières tiennent dans 0.47 point. La chaîne étant déterministe, ces écarts sont exacts ; ils sont mesurés sur un seul jeu de sept machines et ne se transfèrent pas, comme la section 7.2 le montre.

**8.3 La sélection du système retenu utilise le jeu d'évaluation.** Voir section 7.5. Cette information n'existerait pas dans une campagne réelle au moment du choix.

**8.4 Deux stratégies de seuillage s'écartent d'un fonctionnement en ligne.** `otsu` et `mixture` ajustent leur seuil sur la distribution des scores de test. Aucune étiquette n'est utilisée, mais un accès par lot à l'ensemble des données de test est requis. Un déploiement en flux ne le permettrait pas. À déclarer explicitement dans toute comparaison avec `gamma` et `percentile`, qui n'utilisent que les scores d'entraînement.

**8.5 Cinq des sept machines du développement sont émulées.** ToyCarEmu, bearingEmu, gearboxEmu, sliderEmu et valveEmu sont obtenues par convolution de réponses impulsionnelles mesurées. Le contraste proche/lointain de ces machines est donc synthétique dans sa composante spatiale. Deux machines seulement, ToyCar et fan, sont des enregistrements à deux microphones réels. La portée de toute conclusion sur l'exploitation du second canal est limitée d'autant, et cette limite touche directement les hypothèses 1 et 2 de la section 6.

**8.6 Le dossier `supplemental` était indisponible.** Il contient les enregistrements de machine seule et de bruit seul. Son absence empêche de mesurer directement la contamination du canal 2 par la machine, c'est-à-dire de quantifier ce que le masque de la section 6.1 avait à gagner. L'explication du résultat neutre reste donc une hypothèse.

**8.7 L'analyse des rapports d'énergie porte sur un échantillon.** `e0_results/e0b_summary.csv` est calculé sur **60 clips par machine**, tirés au hasard dans le split train source, et non sur les 1000 clips disponibles. Les colonnes `q1_raies_dB` et `modulation_dB`, ainsi que la classification en régime, héritent de cette variance d'échantillonnage, non quantifiée.

**8.8 La branche CNN repose sur une seule graine par condition d'étiquetage.** La sensibilité à la graine vaut 3.25 points, mesurée sur `--pseudo-labels 16`. L'écart apparent entre les deux conditions d'étiquetage vaut 2.08 points, soit moins que cette sensibilité. **L'effet des pseudo-étiquettes n'est donc pas établi**, dans un sens ni dans l'autre. Deux graines ne suffisent pas non plus à estimer une dispersion pour la condition `--pseudo-labels 16`, et la condition `--pseudo-labels 0` n'a qu'une seule graine.

**8.9 La variabilité d'exécution antérieure au correctif n'est pas reproductible.** Les 2.66 points d'écart entre deux exécutions à graine fixée, qui ont motivé le forçage du mode déterministe, ne sont pas vérifiables depuis les fichiers actuels : les exécutions concernées n'ont pas été conservées. Le chiffre est rapporté comme tel en section 7.1. Toutes les mesures CNN de ce rapport proviennent de runs postérieurs au correctif, dont la reproductibilité est vérifiée par duplication.

**8.10 La pAUC est mal résolue.** La région `FPR ≤ 0.1` ne couvre que 10 clips normaux par machine. Le déplacement d'un seul borne la variation de pAUC standardisée à 5.26 points (section 3.6). L'amplitude mesurée reste sous le point, mais aucun écart de pAUC inférieur au point ne doit être interprété.

**8.11 Le journal de campagne est antérieur aux résultats.** Voir section 7.6. Les résultats sont vérifiés indépendamment ; leur journal d'exécution ne l'est pas.

---

## 9. Conclusion et travaux futurs

### 9.1 Ce qui est établi

Trois résultats sont établis par les mesures de ce rapport.

**Le pooling fréquentiel apporte 2.02 points sur le développement.** Conserver les 12 bandes de la grille de tokens d'AST et scorer bande par bande, avec agrégation par maximum, porte le score de 56.54 à 58.56. Aucun entraînement n'est requis.

**La normalisation par domaine transfère favorablement.** Elle coûte 0.65 point sur le développement et en rapporte 4.47 sur l'évaluation. La décomposition montre un échange de 0.89 point d'AUC source contre 10.64 points d'AUC cible sur l'évaluation. La méthode n'utilise aucune étiquette de domaine des clips de test : seule la banque d'entraînement est partitionnée. Elle s'applique donc telle quelle au cadre *first-shot*.

**Le transfert développement → évaluation n'est pas fiable sur ce jeu.** Quatre réglages présentent trois régimes distincts : dégradation constante, retournement favorable, retournement défavorable. Les baselines officielles inversent leur classement entre les deux jeux selon les chiffres publiés. Un écart de moins d'un point sur le développement ne permet aucune décision.

### 9.2 Ce qui est rejeté

Deux voies d'exploitation du second canal ont été construites et rejetées par les mesures.

**Le masque inter-canaux est neutre.** +0.44 sur le développement, −0.08 sur l'évaluation, à configuration strictement appariée. Le contrôle par inversion des canaux, à −7.59 points, confirme que le canal 1 est le canal informatif ; il ne rend pas le masque utile.

**FarMix se retourne.** +0.89 sur le développement, −2.35 sur l'évaluation, avec la perte localisée sur l'AUC cible.

Deux voies annexes ont également été rejetées : le recentrage `--domain-align`, qui dégrade dans les deux jeux, et le CNN entraîné de zéro, qui reste 5.62 à 8.87 points sous la branche AST dans ses trois runs valides. Sur ce dernier, deux régimes défaillants opposés ont été identifiés — mémorisation complète à 98 classes, sous-apprentissage à 38 classes — et la comparaison des deux conditions d'étiquetage n'est pas concluante, l'écart apparent de 2.08 points étant inférieur à la sensibilité à la graine de 3.25 points.

Le bilan est donc net : le bénéfice obtenu dans ce travail vient du *scoring*, non de l'exploitation du second canal. L'objectif de la section 1.3 — obtenir un bénéfice comparable au système gagnant sans simulation ni données externes — n'est pas atteint par la voie visée.

### 9.3 Observation sur le protocole

Le F1, appliqué à un jeu équilibré à 50 % d'anomalies, attribue 66.67 à un système qui déclare tout anormal. Aucune de nos soumissions n'atteint cette valeur ; la meilleure obtient 51.78. La démonstration la plus nette est la case ToyDrone cible de la soumission `align`, qui atteint exactement 66.67 de F1 avec 98 % de rappel pour une AUC de 21.48, c'est-à-dire sur un ordonnancement inversé. Le balayage du quantile est monotone sans optimum, ce qui indique une convergence vers la dégénérescence. Le score officiel, fondé sur AUC et pAUC, n'a pas ce défaut.

### 9.4 Travaux futurs

**Par ordre de priorité pour une publication.**

1. **Rejouer la campagne AST avec le journal conservé.** C'est le dernier écart de traçabilité : les 29 configurations AST sont vérifiées par recalcul, mais leur journal d'exécution date de la campagne précédente. Aucun calcul nouveau au-delà de la campagne elle-même, la branche AST étant déterministe et sans entraînement. La branche CNN, elle, est désormais entièrement documentée par ses cinq journaux.
2. **Étendre les deux conditions d'étiquetage à cinq graines au minimum.** C'est la seule façon de trancher sur les pseudo-étiquettes : l'écart apparent de 2.08 points est aujourd'hui noyé dans une sensibilité à la graine de 3.25 points. Coût : huit runs supplémentaires de 145 s environ, soit vingt minutes de calcul.
3. **Explorer le régime intermédiaire de la branche CNN.** Les deux configurations mesurées encadrent un intervalle non exploré : 38 classes produisent du sous-apprentissage, 98 de la mémorisation complète. Un balayage du nombre de pseudo-classes, ou une réduction du budget d'époques à 98 classes, dirait si un régime utile existe entre les deux. C'est le seul angle par lequel la branche légère garde un intérêt.
4. **Obtenir le dossier `supplemental`** et mesurer la contamination du canal 2 par la machine, machine par machine. C'est le seul moyen de convertir l'hypothèse explicative de la section 6.1 en mesure, et de savoir si le masque échoue parce qu'il n'a rien à gagner ou parce qu'il détruit trop.
5. **Étendre l'analyse de régime aux 1000 clips par machine**, avec intervalle de confiance sur `modulation_dB` et `q1_raies_dB`. Lever la limite 8.7 et tester la corrélation entre régime et gain sur un échantillon complet.
6. **Tester la normalisation par domaine sur des banques de tailles intermédiaires.** L'hypothèse de la section 7.2 prédit que le bénéfice croît avec le déséquilibre. Sous-échantillonner la banque source à 100, 300 et 990 clips fournirait un test direct, sans données nouvelles.
7. **Séparer l'effet de bande de celui de l'agrégation.** Le gain de 2.02 points mélange le pooling par bande et le choix du maximum. Mesurer l'agrégation par quantile intermédiaire entre moyenne et maximum indiquerait où se situe l'optimum et si le maximum est un artefact de ce jeu.
8. **Reprendre le seuillage sur une métrique non dégénérable.** Le balayage monotone du quantile et les deux dégénérescences des sections 7.3 et 7.4 bis sont des symptômes du F1, pas du seuillage. Une métrique de décision qui pénalise le taux de faux positifs — coût asymétrique, ou F1 à précision minimale imposée — permettrait de poser le problème correctement.

**Une réserve pour la publication.** La limite 8.5 est la plus contraignante pour la portée du travail : cinq des sept machines du développement ont un contraste spatial synthétique. Toute affirmation générale sur l'exploitabilité du contraste proche/lointain devrait être restreinte au régime effectivement mesuré, ou appuyée sur les deux machines réelles seulement — ce qui, pour `fan`, ne fournit aucun signal exploitable.

---

*Rapport établi à partir des fichiers du dépôt `dcase2026` dans leur état du 2026-09-03. Toutes les valeurs annotées « Mesure » ont été lues ou recalculées depuis ces fichiers. Les valeurs annotées « Chiffre publié » proviennent de sources tierces et n'ont pas été vérifiées.*
