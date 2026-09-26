# Protocole du smoke test OmniASR CTC-1B

## Objectif

Ce test ne mesure pas encore la qualité finale de transcription. Il vérifie
que l'environnement Kaggle, le dataset publié, la recette officielle, FSDP et
la sauvegarde d'un état d'entraînement complet fonctionnent ensemble sur deux
GPU T4.

## Configuration testée

- modèle : `omniASR_CTC_1B` ;
- dataset : `Tree-AI-lab/baoule-asr-dataset-mixture` ;
- matériel : exactement deux T4 ;
- précision : FP16 ;
- parallélisme : FSDP, deux processus ;
- activation checkpointing : couche par couche ;
- accumulation : huit microbatches ;
- durée maximale temporaire : 20 secondes ;
- entraînement : 20 pas, puis validation et checkpoint.

Le dataset est téléchargé à une révision Hugging Face précise, enregistrée
avec les résultats. Le dépôt Meta est figé au commit
`81f51e224ce9e74b02cc2a3eaf21b2d91d743455`.

## Critères de réussite

Le smoke test est accepté si :

1. le préflight du DataLoader lit deux batches sans erreur ;
2. les deux processus distribués commencent et terminent les 20 pas ;
3. aucune erreur CUDA, perte non finie ou erreur de décodage n'apparaît ;
4. la validation s'exécute ;
5. les shards FSDP du checkpoint sont présents ;
6. la VRAM maximale de chaque T4 laisse une marge exploitable ;
7. l'espace disque restant permet de conserver et transférer le checkpoint.

## Décision après exécution

Les fichiers `smoke_summary.json`, `gpu_metrics.csv` et `smoke_console.log`
seront analysés avant le run suivant. La limite de 40 secondes du vrai
fine-tuning ne sera conservée que si la marge mémoire le permet. Le débit
mesuré déterminera aussi si les évaluations et sauvegardes restent espacées de
500 pas ou doivent être ramenées à 250 pas pour tenir dans une session Kaggle.

Une réussite technique du smoke test n'autorise pas encore l'usage du split
test pour piloter l'entraînement : le choix des checkpoints reposera sur le
split de développement, et le test restera réservé à l'évaluation finale.
