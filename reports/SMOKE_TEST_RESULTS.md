# Résultats du smoke test OmniASR CTC-1B

## Statut

Le smoke test du 26 septembre 2026 est validé. Les 20 pas, la validation et le
checkpoint distribué complet se sont terminés avec un code retour nul sur deux
Tesla T4.

Révisions utilisées :

- OmniASR : `81f51e224ce9e74b02cc2a3eaf21b2d91d743455` ;
- dataset : `Tree-AI-lab/baoule-asr-dataset-mixture` à la révision
  `1aae3edca09e2e7db846afa2904bf648e3a0d83a` ;
- PyTorch : `2.8.0+cu128` ;
- fairseq2 : `0.6` ;
- omnilingual-asr : `0.2.0`.

## Mesures observées

| Mesure | Résultat |
|---|---:|
| Durée totale instrumentée | 417,4 s |
| Durée de la tâche fairseq2 | 377 s |
| Pic `nvidia-smi`, GPU 0 | 14 845 Mio |
| Pic `nvidia-smi`, GPU 1 | 14 905 Mio |
| Pic mémoire active fairseq2 | 13,13 Gio, 91 % |
| Pic mémoire réservée fairseq2 | 14,33 Gio, 99 % |
| Taille de la sortie | 11 669 433 182 octets, environ 10,87 Gio |
| Espace libre après sauvegarde | environ 8,45 Gio |
| Checkpoint | complet : trainer, modèle, optimiseur et DataLoader sur 2 rangs |

La perte CTC publiée est passée de `73,0762` au pas 5 à `46,2487` au pas 20.
La validation au pas 20 a produit un WER de `51,1171` et un UER de `17,0146`
sur les 94 exemples `dev` respectant la limite de 20 secondes.

Deux overflows FP16 sont apparus aux pas 13 et 15. Le scaler dynamique a
ignoré les gradients concernés puis réduit son échelle de 128 à 64 et à 32.
Le run a ensuite terminé normalement.

## Interprétation et décisions

La marge mémoire est trop faible pour doubler directement la durée maximale à
40 secondes. Le premier entraînement réel conserve donc :

- `max_audio_len = 320_000`, soit 20 secondes ;
- `max_num_elements = 320_000` par microbatch ;
- huit microbatches accumulés ;
- FSDP v1, FP16 et activation checkpointing couche par couche.

Cette limite utilise 704 des 806 exemples d'entraînement, soit environ
1,849 heure sur 2,617 heures. Les 102 exemples train plus longs restent dans le
dataset, mais le DataLoader les filtre jusqu'à leur segmentation ultérieure.

L'échelle FP16 initiale est ramenée à 32 pour le vrai run. Le scheduler
tri-stage du smoke test dépendait de ses 20 pas et ne peut pas être repris avec
des cibles successives de 250, 500 et 750 pas sans changer de courbe. Le
pipeline par blocs utilise donc le scheduler `myle` de fairseq2, avec 25 pas de
warm-up ; sa valeur dépend du numéro de pas global et reste continue après une
reprise complète.

La durée de 173,9 minutes affichée par l'extrapolation automatique pour 500 pas
est prudente : elle multiplie aussi les coûts fixes de chargement, validation et
sauvegarde. Le premier bloc est limité à 250 pas. Sa mesure réelle déterminera
la durée des blocs suivants.

Enfin, les checkpoints complets sont trop volumineux pour en copier deux dans
`/kaggle/working`. Lors d'une reprise, le notebook monte le checkpoint
précédent en lecture seule depuis un Input Kaggle, écrit uniquement le nouveau
checkpoint puis détache ce lien avant `Save Version`.
