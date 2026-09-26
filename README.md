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
- [x] Exécution de l'audit complet sur Kaggle
- [x] Analyse statistique initiale de l'audit
- [x] Génération de la file de revue manuelle
- [x] Revue hackathon partielle des exemples signalés
- [x] Politique de sélection hackathon figée et testée
- [x] Script et notebook de préparation MixtureParquet
- [x] Exécution de la préparation MixtureParquet sur Kaggle
- [x] Publication dans `Tree-AI-lab/baoule-asr-dataset-mixture`
- [x] Notebook et lanceur instrumenté du smoke test CTC-1B
- [x] Smoke test CTC-1B sur les deux T4
- [x] Analyse des contraintes mémoire, temps et disque du smoke test
- [x] Pipeline Kaggle d'entraînement et de reprise par blocs de 250 pas
- [ ] Exécution du premier bloc réel de 250 pas
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

## Deuxième étape : revue audio manuelle

Le notebook `notebooks/02_review_audio_kaggle.ipynb` transforme l'audit en
manifeste de décisions, charge les deux datasets aux révisions auditées et
permet d'écouter chaque exemple signalé. Les décisions autorisées sont
`keep`, `trim`, `segment` et `exclude`.

Les deux fichiers minimaux nécessaires à cette étape sont versionnés dans
`baoule_ctc1b_dataset_audit/`. Le notebook les récupère donc avec le dépôt et
fonctionne dans une nouvelle session Kaggle sans ajouter l'audit comme Dataset
d'entrée. Les autres artefacts volumineux de l'audit restent ignorés.

Chaque décision est immédiatement sauvegardée dans
`/kaggle/working/baoule_ctc1b_review/decision_manifest_reviewed.csv`. Le
dossier de revue doit être téléchargé avant la fermeture de la session Kaggle.

## Troisième étape : préparation MixtureParquet

Le notebook `notebooks/03_prepare_mixture_kaggle.ipynb` construit le corpus
utilisé pendant le hackathon. La sélection contient les exemples validés
manuellement, les validations automatiques et les audios Waxal non rejetés de
40 secondes maximum. Les exclusions manuelles sont prioritaires ; les audios
longs et les Klayt non révisés restent différés.

Le script convertit les audios en FLAC mono 16 kHz et crée le partitionnement
Hive requis par OmniASR : `corpus`, `split` et `language=bci_Latn`. Les Parquet
générés restent dans les sorties Kaggle et ne doivent pas être versionnés dans
GitHub.

La dernière cellule du notebook publie le dossier généré dans
`Tree-AI-lab/baoule-asr-dataset-mixture`. Elle lit un token d'écriture depuis
le secret Kaggle `HF_TOKEN` ; le secret n'est jamais stocké dans le dépôt.

## Quatrième étape : smoke test CTC-1B

Le notebook `notebooks/04_smoke_test_ctc1b_kaggle.ipynb` télécharge le dataset
Tree AI Lab, vérifie le DataLoader officiel puis lance 20 pas en FP16 avec FSDP
sur exactement deux T4. Le lanceur `src/run_smoke_test.py` archive :

- toute la sortie console dans `smoke_console.log` ;
- la VRAM et l'utilisation de chaque GPU dans `gpu_metrics.csv` ;
- les versions logicielles, commits, temps, estimations et checkpoints dans
  `smoke_summary.json`.

Le test doit produire un checkpoint complet et terminer sans erreur avant de
préparer le premier bloc de 250 pas. Le dossier de sortie Kaggle doit être
sauvegardé avec **Save Version**, puis téléchargé pour analyse.

Les résultats sont consignés dans
`reports/SMOKE_TEST_RESULTS.md`. Ils ont conduit à retenir des blocs de 250 pas
plutôt que 500 pour la première phase réelle.

## Cinquième étape : entraînement réel par blocs

Le notebook `notebooks/05_train_ctc1b_staged_kaggle.ipynb` exécute le premier
bloc de 0 à 250 pas. Il utilise `configs/ctc_1b_stage.yaml`, valide tous les
50 pas et sauvegarde l'état complet au pas 250.

Après le bloc, un panneau fixe de quatre exemples du split `dev` est transcrit.
Le manifeste du panneau, les prédictions, les métriques, les logs et le
checkpoint sont conservés dans la sortie Kaggle. Pour le bloc suivant, la
sortie précédente doit être ajoutée comme Input et `TARGET_STEP` doit passer à
500. Le checkpoint précédent reste monté en lecture seule afin de ne pas
occuper deux fois environ 10,87 Gio dans `/kaggle/working`.
