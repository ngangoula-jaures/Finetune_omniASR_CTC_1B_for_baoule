# Fine-tuning OmniASR CTC-1B pour le baoulé

Ce dossier contient le travail consacré au fine-tuning de `omniASR_CTC_1B` sur un mélange de :

- `google/WaxalNLP`, configuration `bau_tts` ;
- `Klayt/baoule-common-voice`.

Le protocole complet est décrit dans [PLAN_FINETUNING_OMNIASR_CTC_1B_BAOULE.md](PLAN_FINETUNING_OMNIASR_CTC_1B_BAOULE.md).

## État d'avancement

- [x] Protocole d'entraînement et d'évaluation
- [x] Normalisation textuelle commune aux deux corpus
- [x] Script d'audit automatique des datasets
- [x] Tests unitaires de la normalisation
- [x] Configurations initiales du smoke test et du fine-tuning
- [ ] Exécution de l'audit complet sur Kaggle
- [ ] Revue des exemples signalés par l'audit
- [ ] Préparation du mélange MixtureParquet
- [ ] Smoke test CTC-1B sur les deux T4
- [ ] Entraînement par blocs de 500 pas
- [ ] Évaluation et export Hugging Face

## Organisation

```text
finetuning_omniasr_ctc_1b_baoule/
├── PLAN_FINETUNING_OMNIASR_CTC_1B_BAOULE.md
├── README.md
├── configs/
├── notebooks/
├── reports/
├── src/
└── tests/
```

## Première étape : audit des datasets

Installation minimale :

```bash
python -m pip install -r requirements-audit.txt
```

Exécution :

```bash
python src/audit_datasets.py \
  --output-dir reports/dataset_audit
```

Le script utilise par défaut des révisions figées des deux datasets. Il produit :

- `dataset_audit.csv` : une ligne par audio ;
- `dataset_summary.json` : statistiques par source et split ;
- `character_inventory.json` : inventaire Unicode avant et après normalisation ;
- `duplicate_groups.json` : doublons audio et texte traversant éventuellement les splits ;
- `manual_review.csv` : exemples à écouter ou vérifier avant préparation.

L'audit ne modifie pas les datasets distants et n'entraîne aucun modèle.

### Exécution dans Kaggle avec le dépôt public

Le notebook `notebooks/01_audit_datasets_kaggle.ipynb` clone directement ce dépôt public :

```text
ngangoula-jaures/Finetune_omniASR_CTC_1B_for_baoule
```

Avant de l'exécuter :

1. pousser les fichiers locaux vers GitHub ;
2. vérifier que le dépôt est public ;
3. activer Internet dans les options de la session Kaggle ;
4. exécuter les cellules du notebook dans l'ordre.

Aucun token GitHub ni secret Kaggle n'est nécessaire pour le clonage en lecture seule.

## Tests locaux

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tests
```

## Règle avant entraînement

Le mélange MixtureParquet ne sera produit qu'après lecture du résumé d'audit et décision sur les exemples contenant une voix parasite, une transcription douteuse, un doublon ou un fichier trop long.
