# Configurations CTC-1B

Ces fichiers ciblent deux T4 Kaggle et la révision OmniASR utilisée par le benchmark.

Ils supposent que la phase de préparation créera :

```text
/kaggle/working/baoule_ctc1b_data/baoule_mixed/version=0
/kaggle/working/baoule_ctc1b_data/language_distribution_0.tsv
/kaggle/working/baoule_ctc1b_data/assets/baoule_mixed.yaml
```

La carte asset `baoule_mixed.yaml` référence le chemin des Parquet et permet à
la recette de résoudre le dataset par son nom `baoule_mixed`.

La configuration `smoke` doit réussir avant la configuration de 5 000 pas. Les valeurs de `max_audio_len`, `max_num_elements` et `grad_accumulation` seront ajustées après mesure réelle de la mémoire sur les deux T4.

Les deux fichiers utilisent FSDP. Ils doivent être lancés avec deux processus distribués ; un processus unique ou du DDP classique ne fournit pas le même sharding mémoire.

`ctc_1b_smoke.yaml` limite les audios à 20 secondes et exécute 20 pas.
`ctc_1b_stage.yaml` est la configuration retenue après le smoke test : limite
de 20 secondes, scheduler `myle`, validations tous les 50 pas et checkpoints
complets tous les 250 pas. Le notebook remplace seulement `num_steps` pour
atteindre successivement 250, 500, 750 pas, etc.

`ctc_1b_finetune.yaml` reste une configuration expérimentale de référence à
5 000 pas et 40 secondes. Elle ne doit pas être lancée sur les deux T4 : le
smoke test a montré que la mémoire était déjà réservée à 99 % avec 20 secondes.
