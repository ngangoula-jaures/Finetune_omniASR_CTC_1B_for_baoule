# Configurations CTC-1B

Ces fichiers ciblent deux T4 Kaggle et la révision OmniASR utilisée par le benchmark.

Ils supposent que la phase de préparation créera :

```text
/kaggle/working/baoule_ctc1b_data/version=0
/kaggle/working/baoule_ctc1b_data/language_distribution_0.tsv
/kaggle/working/baoule_ctc1b_assets/baoule_mixed.yaml
```

La configuration `smoke` doit réussir avant la configuration de 5 000 pas. Les valeurs de `max_audio_len`, `max_num_elements` et `grad_accumulation` seront ajustées après mesure réelle de la mémoire sur les deux T4.

Les deux fichiers utilisent FSDP. Ils doivent être lancés avec deux processus distribués ; un processus unique ou du DDP classique ne fournit pas le même sharding mémoire.

