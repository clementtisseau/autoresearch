# autoresearch-ladder : valider un critère « loi d'échelle »

Fork expérimental d'autoresearch. Une idée n'est plus jugée sur un seul run de 5 min (critère **A**),
mais sur 3 petits runs (nano, micro, mini), dont on extrapole la loss à ~10× le compute (critère **B**).
Avant de brancher B dans la boucle de l'agent, on vérifie qu'il prédit mieux qu'A le résultat d'un vrai run 10×.

## Fichiers

| Fichier | Rôle | Modifiable par l'agent |
|---|---|---|
| `train.py` | Inchangé, sauf : taille, budget de FLOPs, seed et logs passés par variables d'environnement | oui |
| `ladder_lib.py` | Comptage des FLOPs (C = 6·N·D, N sans embeddings), ordre des données par seed, logs | non |
| `ladder.py` | Lance les runs, ajuste la loi, calcule A, B et les verdicts | non |
| `sizes.py` → `ladder_config.json` | Tailles et budgets des 4 échelons (calculés une fois, puis figés) | non |

## Les échelons (`uv run sizes.py`)

| Échelon | depth | width | compute / mini | tokens | ~durée H100 |
|---|---|---|---|---|---|
| nano  | 6  | 384 | 0,22 | 234M  | ~1 min |
| micro | 8  | 384 | 0,35 | 295M  | ~1,5 min |
| mini  | 8  | 512 | 1    | 500M  | ~4-5 min (= baseline Karpathy) |
| big   | 12 | 768 | 9,65 | 1,55B | ~40-50 min |

Le ratio tokens/paramètres est identique (17) à tous les échelons. Les budgets sont en FLOPs, pas en secondes.

## Critères

- **A** : val_bpb de mini.
- **B** : ajustement `log(L − E) = a + b·log C` sur nano/micro/mini **avec pente libre par idée**,
  prédiction de L au compute de big. `--E auto` estime E sur le baseline (fiable une fois les runs big du baseline faits).
- **B_offset** (diagnostic) : même chose, avec la pente du baseline imposée.
- Verdict « keep » si `gain > 2 · σ · √(1/n_idée + 1/n_baseline)`, avec σ mesuré sur les seeds du baseline.

## Protocole (les portes)

```bash
uv run prepare.py --num-shards 40        # big consomme ~1,6B tokens : plus de shards que par défaut
uv run ladder.py run baseline --seeds 0 1 2                        # porte 1 (~0,5 h)
uv run ladder.py noise                                             # σ_A, σ_B : B est-il trop bruité ?
uv run ladder.py run <idee> --patch ideas/<idee>.patch             # porte 2, pour chaque idée
uv run ladder.py report                                            # verdicts A/B, désaccords
uv run ladder.py run baseline --rungs big --seeds 0 1              # porte 3 (~1,5 h)
uv run ladder.py run <idee> --patch ideas/<idee>.patch --rungs big # porte 4, désaccords d'abord
uv run ladder.py report --E auto                                   # Spearman, décisions justes, faux keeps
```

Les résultats vont dans `runs/<tag>/` (un `.json`, un `.log` et un `.steps.jsonl` avec loss et norme du gradient à chaque pas).
Relancer une commande saute les runs déjà faits : on peut interrompre et reprendre.

## Tests (CPU, sans GPU)

```bash
python -m pytest tests -q
```

`tests/fake_train.py` simule une loi d'échelle, avec une « bonne » idée et une idée piège (elle aide en petit et nuit en grand).
Le chemin GPU de `train.py` (boucle d'entraînement, FA3) n'a pas encore été exécuté.
