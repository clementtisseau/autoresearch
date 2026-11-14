# Idées pour la porte 2

15 patchs de `train.py`, tous tirés de runs réels. Régénérer avec `python ideas/make_ideas.py`.

**Pièges présumés** : gardés dans des runs courts ou petits, mais transfert douteux à plus grande échelle.

| Patch | Changement | Source |
|---|---|---|
| `halve_batch` | batch total 524K → 262K tokens | run de Karpathy (gardé) |
| `seed_change` | décalage de seed (42 → 137) : idée de pur bruit | run de Karpathy (gardé), discussion #285 |
| `window_1_8` | fenêtre courte = 1/8 du contexte (256 tokens) | run de Karpathy (gardé) ; nanochat a finalement retenu ~1/3 |
| `window_1_4` | fenêtre courte = 1/4 du contexte | run de Karpathy (gardé) |
| `window_sssssl` | motif SSSSL (moins de couches à attention complète) | run de Karpathy (gardé) |
| `warmup_5pct` | warmup de 5 % | run de Karpathy (gardé) |
| `adam_beta2_999` | Adam beta2 0,95 → 0,999 | issue #596, RTX 3060 (gardé) |
| `scalar_lr_03` | SCALAR_LR 0,5 → 0,3 | issue #596, RTX 3060 (gardé) |

**Témoins** : selon Karpathy, ces changements tiennent de depth 12 à depth 24 (commit nanochat `6ed7d1d`).

| Patch | Changement | Source |
|---|---|---|
| `unembed_lr_x2` | LR de l'unembedding 0,004 → 0,008 | nanochat `6ed7d1d` + run de Karpathy |
| `rope_100k` | base RoPE 10K → 100K | nanochat `6ed7d1d` + run de Karpathy |
| `weight_decay_028` | weight decay 0,2 → 0,28 | nanochat `6ed7d1d` |
| `warmdown_065_final_005` | warmdown 0,5 → 0,65, LR finale 0 → 5 % | nanochat `6ed7d1d` |
| `embed_init_08` | init des embeddings std 1,0 → 0,8 | nanochat `6ed7d1d` |
| `muon_momentum` | Muon beta2 0,95 → 0,9, momentum 0,95 → 0,97 sur 400 pas | nanochat `6ed7d1d` |
| `x0_lambda_005` | init x0_lambda 0,1 → 0,05 | run de Karpathy (gardé) |

À noter :
- Les étiquettes « piège » et « témoin » sont des hypothèses : c'est justement le run big qui tranche.
- Les idées qui changent la taille du modèle (depth, aspect ratio) sont exclues, car la taille est imposée par la ladder.

## Lancer la porte 2

```bash
for p in ideas/*.patch; do uv run ladder.py run "$(basename "$p" .patch)" --patch "$p"; done
```
```bash
uv run ladder.py report
```
