# Rapport complet du projet de fine-tuning OmniASR CTC-1B pour le baoulé

**Organisation :** Tree AI Lab  
**Projet applicatif :** Kouman AI  
**État du rapport :** 26 septembre 2026  
**Statut actuel :** données préparées et publiées, smoke test validé, pipeline réel prêt ; premier bloc de 250 pas non encore exécuté.

## 1. Résumé exécutif

Tree AI Lab développe Kouman AI, une intelligence artificielle destinée à comprendre et parler des langues africaines locales. La première langue prise en charge est le baoulé. Pour rendre le système accessible oralement, notamment dans un contexte où une partie des utilisateurs lit peu ou pas, l'architecture retenue est une cascade :

```text
parole en baoulé → ASR → modèle Kouman AI → TTS → réponse orale
```

Le TTS et le modèle central sont déjà fonctionnels. Le principal point faible identifié est l'ASR, c'est-à-dire la transformation de la parole en texte.

Un premier ASR avait été obtenu en adaptant `facebook/omniASR-CTC-300M` sur Waxal pendant 1 000 pas. Ce modèle, publié sous `Tree-AI-lab/omniASR-CTC-300M-baoule-1000steps`, a apporté une amélioration, mais sa précision et sa robustesse restaient insuffisantes pour Kouman AI.

Un benchmark séparé de plusieurs modèles OmniASR a ensuite conduit à retenir `omniASR_CTC_1B` comme nouvelle base principale. Le modèle LLM 300M reste une piste d'expérimentation ultérieure. Le présent dépôt a donc été créé pour reconstruire un protocole de fine-tuning plus rigoureux autour du CTC-1B, avec :

- deux sources audio complémentaires au lieu d'une seule ;
- un audit reproductible des données ;
- une revue humaine des alignements difficiles ;
- une normalisation textuelle adaptée au baoulé ;
- un dataset au format officiel OmniASR ;
- un entraînement distribué compatible avec deux GPU T4 Kaggle ;
- des validations et transcriptions intermédiaires ;
- une reprise exacte de l'entraînement entre plusieurs sessions Kaggle.

Le projet est arrivé au seuil du véritable entraînement : le dataset a été créé et envoyé sur Hugging Face, le test technique de 20 pas a réussi, et le notebook d'entraînement par blocs de 250 pas est prêt.

## 2. Objectifs et critères de réussite

L'objectif n'est pas seulement d'entraîner un modèle plus grand. Il faut produire un ASR baoulé qui soit :

- plus précis que le précédent CTC-300M Tree AI Lab ;
- robuste aux différences de voix, de microphones et de bruit ;
- compatible avec l'usage oral de Kouman AI ;
- évalué sur des données indépendantes de l'entraînement ;
- reproductible grâce à des versions, configurations et checkpoints identifiables ;
- entraînable malgré les limites des sessions Kaggle T4 x2.

Une amélioration devra être établie sur des métriques comparables, principalement le WER et le CER, mais aussi par l'écoute et la lecture de transcriptions réelles. Le split `dev` sert aux décisions pendant l'entraînement. Le split `test` doit rester gelé jusqu'à l'évaluation finale.

### Définitions utiles

- **ASR** : système qui transforme un audio parlé en texte.
- **CTC** : objectif d'entraînement permettant d'apprendre l'alignement entre un signal audio et une transcription sans horodatage mot à mot.
- **Fine-tuning** : adaptation d'un modèle déjà préentraîné à une langue ou à un domaine précis.
- **WER** : taux d'erreur sur les mots. Plus il est faible, meilleur est le résultat.
- **CER** : taux d'erreur sur les caractères, utile pour analyser les erreurs orthographiques fines.
- **UER** : taux d'erreur sur les unités produites par le tokenizer du modèle.
- **Checkpoint** : sauvegarde du modèle et, ici, de tout l'état nécessaire pour reprendre exactement l'entraînement.
- **FSDP** : technique qui répartit un grand modèle et ses états sur plusieurs GPU afin de réduire la mémoire utilisée par chacun.
- **MixtureParquet** : format de données partitionné attendu par la recette officielle OmniASR.

## 3. Historique du projet

### 3.1 Premier modèle Tree AI Lab

Le point de départ était `facebook/omniASR-CTC-300M`, qui savait déjà transcrire le baoulé mais de manière imparfaite. Tree AI Lab l'a adapté avec Waxal, ce qui a donné `Tree-AI-lab/omniASR-CTC-300M-baoule-1000steps`.

Ce premier travail a validé la faisabilité, mais plusieurs limites ont été reconnues : une seule source de données, 1 000 pas au maximum, peu de diversité acoustique, absence d'audit systématique des doublons et des fuites, évaluation trop réduite, absence d'augmentation audio explicite et sauvegarde insuffisante pour une reprise exacte.

### 3.2 Benchmark des modèles OmniASR

Un benchmark distinct a comparé les familles W2V, CTC et LLM de plusieurs tailles, ainsi que le modèle Tree AI Lab. À l'issue de cette phase, le choix de travail est devenu :

- `omniASR_CTC_1B` comme base à fine-tuner maintenant ;
- le CTC-300M Tree AI Lab comme référence historique ;
- `omniASR_LLM_300M` comme expérience future.

Le benchmark n'est pas contenu dans ce dépôt. Ce dépôt commence à la phase suivante : préparation et fine-tuning du CTC-1B.

### 3.3 Nouvelle stratégie de données

Deux corpus ont été combinés :

1. `google/WaxalNLP`, configuration `bau_tts`, constitué d'enregistrements de bonne qualité mais comprenant de nombreux fichiers longs ;
2. `Klayt/baoule-common-voice`, plus court et plus difficile acoustiquement, avec du bruit, des silences et parfois de la parole parasite avant le traducteur.

Waxal apporte la qualité ; Klayt apporte des conditions plus réalistes. Le mélange doit cependant préserver l'alignement audio–texte : du bruit non vocal peut être utile, mais des mots prononcés et absents de la transcription rendent l'objectif CTC contradictoire.

### 3.4 Adaptation au délai du hackathon

La revue humaine complète de 2 092 exemples aurait demandé trop de temps. Une politique provisoire a donc été figée :

- conserver les décisions humaines déjà enregistrées ;
- conserver les validations automatiques ;
- ajouter les audios Waxal non rejetés de 40 secondes ou moins ;
- différer les Klayt non révisés ;
- différer les audios nécessitant une segmentation ;
- ne jamais annuler une exclusion humaine.

Cette version est adaptée au hackathon, mais ne constitue pas le dataset final de Tree AI Lab.

## 4. Données, révisions et résultats de l'audit

Les sources ont été figées pour rendre l'expérience reproductible :

| Source | Configuration | Révision auditée |
|---|---|---|
| `google/WaxalNLP` | `bau_tts` | `5f4d8ca24f2b9d168b2ee545f1febaaff4b40580` |
| `Klayt/baoule-common-voice` | configuration par défaut | `96eb4413b7cc3190c39a35ee03eccb321c2b9929` |

L'audit complet a parcouru les trois splits des deux corpus :

| Source | Split | Exemples | Locuteurs | Durée |
|---|---:|---:|---:|---:|
| Klayt | train | 319 | 3 | 0,694 h |
| Klayt | validation | 267 | 3 | 0,506 h |
| Klayt | test | 290 | 9 | 0,599 h |
| Waxal | train | 972 | 4 | 10,975 h |
| Waxal | validation | 122 | 4 | 1,483 h |
| Waxal | test | 122 | 4 | 1,596 h |
| **Total** | — | **2 092** | — | **15,854 h** |

Résultats automatiques :

- les 2 092 fichiers ont été décodés ;
- 1 480 ont été acceptés sans signalement ;
- 612 ont été signalés pour revue ;
- aucun rejet technique automatique ;
- aucun doublon audio exact ;
- cinq groupes de textes normalisés dupliqués ;
- Waxal est entièrement en 48 kHz et doit être rééchantillonné à 16 kHz ;
- Klayt est déjà en 16 kHz ;
- 262 fichiers Waxal dépassent 35 secondes et représentent une grande partie de la durée totale ;
- les locuteurs Klayt sont disjoints entre train, validation et test ;
- les mêmes quatre locuteurs Waxal sont présents dans les trois splits.

Les principaux motifs de revue sont les audios trop longs, les silences de bord, les niveaux sonores faibles, les ratios texte/durée anormaux et une fuite textuelle Waxal entre train et test.

## 5. Revue humaine et sélection hackathon

La file de revue générée contient 1 420 exemples. Elle inclut non seulement les 612 signalements automatiques, mais aussi les Klayt apparemment propres, car un simple détecteur de silence ne sait pas identifier une autre voix non transcrite.

Au moment de figer la version hackathon :

- 383 éléments de la file avaient reçu une décision ;
- 1 037 restaient à examiner ;
- le manifeste complet comptait 934 `keep`, 68 `segment`, 53 `exclude` et 1 037 décisions en attente.

Les décisions signifient :

- `keep` : conserver le couple audio–texte ;
- `trim` : enlever uniquement le silence ou la portion hors transcription avec des bornes vérifiées ;
- `segment` : différer le fichier jusqu'à ce qu'un alignement permette d'associer chaque segment au bon texte ;
- `exclude` : ne pas utiliser l'exemple ;
- vide : décision encore en attente.

Le dataset hackathon final contient :

| Source | Split final | Exemples | Durée approximative |
|---|---:|---:|---:|
| Klayt | train | 23 | 0,061 h |
| Klayt | dev | 15 | 0,034 h |
| Klayt | test | 212 | 0,440 h |
| Waxal | train | 783 | 2,556 h |
| Waxal | dev | 95 | 0,326 h |
| Waxal | test | 91 | 0,288 h |
| **Total** | — | **1 219** | **3,706 h** |

Le train contient 806 exemples et environ 2,617 heures. Sur les 2 092 exemples audités, 1 219 ont été sélectionnés, 820 différés et 53 exclus.

Le dataset produit a été publié sous `Tree-AI-lab/baoule-asr-dataset-mixture`. Lors du smoke test, la révision utilisée était `1aae3edca09e2e7db846afa2904bf648e3a0d83a`.

## 6. Prétraitement appliqué

### Texte

La normalisation est volontairement conservatrice :

- conversion Unicode en NFC ;
- passage en minuscules ;
- harmonisation de plusieurs apostrophes vers `'` ;
- suppression des annotations entre crochets ;
- conservation des lettres, marques diacritiques et chiffres intégrés à un mot ;
- suppression des nombres isolés et de la ponctuation non prononcée ;
- normalisation des espaces ;
- conservation des lettres baoulé et des tons.

Les tons ne sont pas retirés, car ils peuvent porter une information linguistique.

### Audio

Chaque audio sélectionné est :

- décodé dans son format d'origine ;
- converti en mono si nécessaire ;
- rééchantillonné à 16 kHz ;
- rogné seulement si une décision `trim` fournit des bornes valides ;
- encodé en FLAC PCM 16 bits ;
- rejeté par la matérialisation s'il devient inférieur à une seconde ou supérieur à 40 secondes.

La sortie suit un partitionnement Hive par `corpus`, `split` et `language=bci_Latn`. Les lignes Parquet contiennent `text`, `audio_bytes` et `audio_size`.

## 7. Contraintes Kaggle et smoke test

Le matériel disponible est une session Kaggle avec deux Tesla T4 d'environ 15 Gio chacune, 30 Gio de RAM et un stockage de travail limité. L'environnement compatible retenu est :

- PyTorch `2.8.0+cu128` ;
- torchaudio `2.8.0+cu128` ;
- fairseq2 `0.6` ;
- omnilingual-asr `0.2.0` ;
- dépôt Meta au commit `81f51e224ce9e74b02cc2a3eaf21b2d91d743455`.

Le smoke test distribué de 20 pas a réussi :

| Mesure | Résultat |
|---|---:|
| Code retour | 0 |
| Durée instrumentée | 417,4 s |
| Durée de la tâche fairseq2 | 377 s |
| Pic GPU 0 | 14 845 Mio |
| Pic GPU 1 | 14 905 Mio |
| Mémoire réservée fairseq2 | 14,33 Gio, soit 99 % |
| Taille du checkpoint | environ 10,87 Gio |
| Espace libre après sauvegarde | environ 8,45 Gio |
| WER dev au pas 20 | 51,1171 |
| UER dev au pas 20 | 17,0146 |

La perte CTC est passée de 73,0762 au pas 5 à 46,2487 au pas 20. Deux overflows FP16 ont eu lieu aux pas 13 et 15 ; le scaler dynamique est descendu de 128 à 64 puis 32, et l'exécution s'est terminée normalement.

Ces mesures ont imposé les décisions suivantes :

- ne pas utiliser la configuration 40 secondes sur les T4 ;
- limiter entraînement et validation à 20 secondes, soit 320 000 échantillons ;
- utiliser FSDP v1, FP16 et l'activation checkpointing couche par couche ;
- accumuler huit microbatches ;
- démarrer le loss scale FP16 à 32 ;
- sauvegarder par blocs de 250 pas ;
- monter l'ancien checkpoint depuis `/kaggle/input` au lieu de le recopier dans `/kaggle/working`.

Avec la limite de 20 secondes, 704 des 806 exemples train restent utilisables, soit environ 1,849 heure sur 2,617 heures. Les 102 autres ne sont pas supprimés du dataset ; ils sont seulement filtrés pendant cette phase et pourront être récupérés après segmentation.

## 8. Entraînement réel prévu

Le pipeline est organisé en blocs successifs : `0→250`, `250→500`, `500→750`, etc. Chaque bloc :

1. télécharge exactement une révision du dataset ;
2. vérifie le DataLoader officiel ;
3. reprend, si nécessaire, l'état complet du bloc précédent ;
4. entraîne jusqu'au pas cible ;
5. valide tous les 50 pas ;
6. publie les métriques tous les 25 pas ;
7. sauvegarde un checkpoint complet au dernier pas du bloc ;
8. transcrit le même panneau de quatre audios `dev` ;
9. exige une analyse humaine avant d'autoriser le bloc suivant.

Le scheduler `myle` utilise 25 pas de warm-up, un LR initial de `1e-7` et un LR d'optimiseur de `1e-5`. Il a été choisi parce que sa trajectoire dépend du pas global et reste cohérente lorsque la cible passe de 250 à 500. Le scheduler tri-stage du smoke test recalculerait sa courbe si la cible finale changeait entre deux sessions.

Un checkpoint complet conserve les états du trainer, du modèle, de l'optimiseur et du DataLoader sur les deux rangs. La reprise ne se contente donc pas de recharger les poids : elle poursuit le même entraînement.

Le premier bloc de 250 pas n'a pas encore été exécuté à la date de ce rapport.

## 9. Organisation du dépôt

```text
finetuning_omniasr_ctc_1b_baoule/
├── README.md
├── RAPPORT_COMPLET_PROJET.md
├── PLAN_FINETUNING_OMNIASR_CTC_1B_BAOULE.md
├── requirements-audit.txt
├── requirements-data.txt
├── baoule_ctc1b_dataset_audit/
├── baoule_ctc1b_review/
├── configs/
├── notebooks/
├── reports/
├── src/
└── tests/
```

### Fichiers racine

- `.gitignore` : empêche de versionner les caches, environnements, sorties volumineuses, checkpoints et artefacts Kaggle inutiles.
- `README.md` : guide opérationnel court et état d'avancement courant. Il reste distinct du présent rapport.
- `RAPPORT_COMPLET_PROJET.md` : document de référence complet du projet.
- `PLAN_FINETUNING_OMNIASR_CTC_1B_BAOULE.md` : plan scientifique et technique détaillé établi avant l'implémentation. Il explique les risques de l'ancien entraînement, les règles de données, l'évaluation, les augmentations envisagées, l'arrêt anticipé, la reproductibilité et les livrables.
- `requirements-audit.txt` : dépendances minimales pour auditer les datasets (`datasets`, NumPy, soundfile).
- `requirements-data.txt` : dépendances nécessaires à la préparation et à la publication du MixtureParquet (`datasets`, Hub Hugging Face, pandas, PyArrow, SciPy, soundfile, etc.).

## 10. Description de chaque notebook et de chaque cellule

Les notebooks sont numérotés dans leur ordre normal d'exécution. Chaque notebook Kaggle peut utiliser une nouvelle session ; les informations persistantes sont récupérées depuis GitHub, Hugging Face ou une sortie Kaggle ajoutée comme Input.

### `notebooks/01_audit_datasets_kaggle.ipynb` — 9 cellules

1. **Introduction** : précise que le notebook audite seulement les données et ne lance aucun entraînement.
2. **Dépendances** : installe les bibliothèques minimales de décodage et d'analyse.
3. **Dépôt du projet** : clone le dépôt GitHub et vérifie la présence du script d'audit.
4. **Titre du test réduit** : annonce la vérification sur deux exemples par split.
5. **Smoke test des données** : exécute l'audit réduit et affiche son résumé pour détecter rapidement un problème de téléchargement ou décodage.
6. **Titre de l'audit complet** : rappelle que les splits test restent séparés.
7. **Audit complet** : traite les trois splits de Waxal et Klayt aux révisions figées.
8. **Lecture des résultats** : affiche les statistiques par groupe, les totaux et les 100 premiers exemples à revoir.
9. **Sauvegarde** : indique quel dossier Kaggle conserver ou télécharger.

### `notebooks/02_review_audio_kaggle.ipynb` — 12 cellules

1. **Introduction** : présente la revue et les quatre décisions possibles.
2. **Dépendances** : installe datasets, pandas, ipywidgets et les bibliothèques audio.
3. **Dépôt du projet** : clone ou met à jour le dépôt et vérifie le script de file de revue.
4. **Recherche de l'audit** : décrit l'ordre de recherche des artefacts.
5. **Résolution du dossier** : cherche d'abord dans `/kaggle/working`, puis dans le dépôt, puis dans `/kaggle/input`; `AUDIT_DIR_OVERRIDE` permet un chemin explicite.
6. **Création/restauration des manifests** : prépare le dossier de revue, restaure éventuellement une sauvegarde et génère la file priorisée.
7. **Titre du chargement audio** : explique l'utilisation des révisions déjà auditées.
8. **Chargement des corpus** : télécharge Waxal et Klayt avec `Audio(decode=False)`.
9. **Consignes de revue** : explique les bornes de rognage, la segmentation et les notes.
10. **Interface interactive** : affiche métadonnées, textes et lecteur audio ; filtre par source/split ; enregistre immédiatement chaque décision, les notes et les bornes.
11. **Consigne d'export** : rappelle de sauvegarder avant la fin de la session.
12. **Archive** : écrit la progression et crée un ZIP téléchargeable.

### `notebooks/03_prepare_mixture_kaggle.ipynb` — 13 cellules

1. **Introduction** : définit la sélection hackathon.
2. **Dépendances** : installe les outils de préparation, rééchantillonnage et Parquet.
3. **Dépôt et manifeste** : clone/met à jour le projet, puis vérifie le script et le manifeste révisé.
4. **Titre de la prévisualisation** : annonce les nombres attendus avant téléchargement audio.
5. **Sélection seule** : applique les règles et exige exactement 1 219 sélectionnés, 820 différés et 53 exclus.
6. **Titre de la construction** : décrit la conversion en mono 16 kHz FLAC et le format Hive.
7. **Matérialisation** : télécharge les sources, traite les audios et écrit le MixtureParquet sans écraser une sortie existante.
8. **Titre des contrôles** : annonce la vérification structurelle et audio.
9. **Validation locale** : contrôle schéma, 1 219 lignes, langue, splits, durées et décodage de plusieurs FLAC ; vérifie aussi la carte du dataset.
10. **Conservation Kaggle** : recommande `Save Version` et interdit de placer les Parquet volumineux dans GitHub.
11. **Instructions Hugging Face** : explique le secret `HF_TOKEN` et la visibilité privée par défaut.
12. **Publication** : authentifie le compte, crée le dépôt dataset et envoie le dossier vers `Tree-AI-lab/baoule-asr-dataset-mixture`.
13. **Vérification distante** : contrôle les fichiers obligatoires et la présence des Parquet sur Hugging Face.

### `notebooks/04_smoke_test_ctc1b_kaggle.ipynb` — 15 cellules

1. **Introduction** : définit le test distribué de 20 pas et la limite de 20 secondes.
2. **Consigne PyTorch** : demande T4 x2, Internet et un redémarrage si PyTorch doit être remplacé.
3. **Compatibilité PyTorch** : impose PyTorch/torchaudio 2.8.0 CUDA 12.8.
4. **Titre de l'installation** : explique le gel de la révision Meta.
5. **Installation complète** : clone OmniASR et le projet, checkout le commit Meta et installe fairseq2/OmniASR.
6. **Contrôle matériel/logiciel** : exige exactement deux GPU et affiche versions et `nvidia-smi`.
7. **Titre du dataset** : demande le secret Hugging Face.
8. **Téléchargement reproductible** : résout le SHA distant, télécharge le dataset et écrit un fichier de révision.
9. **Titre du préflight** : annonce deux itérations du DataLoader officiel.
10. **Préflight** : vérifie partitions, FLAC, tokenizer et batching avant de charger le modèle 1B.
11. **Titre du run** : explique les mesures et l'exigence d'une sortie vide.
12. **Lancement** : valide la configuration, vérifie au moins 18 Gio libres et exécute le runner distribué.
13. **Titre de la validation** : définit les critères de réussite du checkpoint.
14. **Contrôle final** : lit le résumé, exige un code 0 et un checkpoint complet au pas 20, puis affiche la fin du log.
15. **Sauvegarde** : demande `Save Version` et une analyse avant tout entraînement long.

### `notebooks/05_train_ctc1b_staged_kaggle.ipynb` — 22 cellules

1. **Introduction** : présente les blocs de 250 pas, les validations et le panneau `dev`.
2. **Titre PyTorch**.
3. **Compatibilité PyTorch** : installe 2.8.0/cu128 si nécessaire et impose un redémarrage propre.
4. **Titre de l'installation**.
5. **Dépôts/dépendances** : installe les versions figées et récupère le dernier projet.
6. **Contrôle T4 x2** : vérifie versions, nombre de GPU et mémoire disponible.
7. **Titre du dataset**.
8. **Dataset exact** : résout et télécharge la révision Hugging Face utilisée pour le bloc.
9. **Titre du préflight**.
10. **Préflight DataLoader** : lit une itération avant l'entraînement coûteux.
11. **Consigne du bloc** : indique `TARGET_STEP=250` au premier run puis 500 au suivant.
12. **Configuration dynamique** : calcule `START_STEP`, copie le YAML et ne change que `num_steps`; vérifie le scheduler, la limite audio et le dossier stable.
13. **Titre de la reprise**.
14. **Préparation de la sortie** : crée un dossier vide ou trouve le checkpoint antérieur dans `/kaggle/input`, le monte par lien symbolique, recopie le panneau `dev` et exige 16 Gio libres.
15. **Titre du lancement**.
16. **Entraînement distribué** : appelle le runner avec pas initial, pas cible, dataset et révision.
17. **Titre du contrôle checkpoint**.
18. **Validation/détachement** : exige un checkpoint cible complet, puis retire le lien vers l'ancien checkpoint avant `Save Version`.
19. **Titre du panneau témoin**.
20. **Transcription** : charge le checkpoint et transcrit quatre exemples `dev` fixes.
21. **Résultats lisibles** : affiche WER/CER globaux, références, hypothèses et erreurs par exemple.
22. **Décision humaine** : demande de sauvegarder et d'analyser WER, overflows et transcriptions avant d'augmenter la cible de 250.

## 11. Description des configurations

### `configs/README.md`

Documente les chemins Kaggle attendus, le rôle des trois YAML, l'obligation d'utiliser deux processus FSDP et l'interdiction pratique d'utiliser la configuration 40 secondes sur les T4 mesurées.

### `configs/ctc_1b_smoke.yaml`

Configuration validée sur 20 pas : CTC-1B, MixtureParquet, audios de 1 à 20 secondes, microbatch maximal de 320 000 échantillons, LR `1e-5`, scheduler tri-stage, FSDP, FP16, accumulation 8, activation checkpointing, validation et checkpoint au pas 20.

### `configs/ctc_1b_stage.yaml`

Configuration réelle retenue : mêmes limites mémoire, loss scale initial 32, scheduler `myle`, warm-up de 25 pas, métriques tous les 25 pas, validation tous les 50 pas et checkpoint tous les 250 pas. `no_sweep_dir: true` garantit que les reprises retrouvent le même dossier malgré le changement de cible.

### `configs/ctc_1b_finetune.yaml`

Ancienne configuration expérimentale de référence : 5 000 pas, audios de 40 secondes, scheduler tri-stage et checkpoints tous les 500 pas. Elle **ne doit pas être lancée sur les deux T4 actuelles**, car le smoke test atteint déjà 99 % de mémoire réservée à 20 secondes.

## 12. Description des scripts `src`

### `src/__init__.py`

Déclare `src` comme paquet Python et décrit son rôle général.

### `src/normalize_text.py`

Centralise la normalisation des transcriptions. Il harmonise Unicode et les apostrophes sans supprimer les tons, produit l'inventaire de caractères et décrit les points de code Unicode. Le même traitement est utilisé pour préparer les données et calculer les métriques du panneau.

### `src/audit_datasets.py`

Charge les deux datasets à des révisions figées, décode chaque audio, mesure durée, canaux, fréquence, pic, RMS, écrêtage, silences de bord et ratio mots/seconde. Il calcule des empreintes SHA-256 audio/texte, repère les doublons, normalise les textes et classe chaque ligne en `accept`, `review` ou `reject`. Il génère les CSV/JSON de l'audit sans modifier les datasets distants.

### `src/create_review_queue.py`

Transforme l'audit complet en manifeste et file priorisée. Les erreurs techniques et fuites train-test passent en priorité 0, les Klayt signalés en priorité 1, les autres Klayt en priorité 2, les fichiers longs/doublons en priorité 3 et les silences rognables en priorité 4. Il ajoute les champs éditables nécessaires à la revue.

### `src/prepare_mixture.py`

Applique les décisions humaines et la politique hackathon. Il peut d'abord simuler la sélection sans télécharger les données. En mode complet, il décode, convertit, rééchantillonne, rogne et encode les audios, écrit les Parquet par groupes, produit la distribution linguistique, la carte asset OmniASR, le manifeste final, le résumé et une fiche dataset.

### `src/run_smoke_test.py`

Lance la recette officielle avec `torch.distributed.run` sur deux GPU. En parallèle, il interroge `nvidia-smi` et écrit l'utilisation mémoire, GPU, température et puissance. Il archive la console, inspecte les shards du checkpoint, mesure le temps et le disque, enregistre les versions et crée `smoke_summary.json`.

### `src/run_training_stage.py`

Lance un bloc réel. Il vérifie le checkpoint initial attendu, empêche de mélanger deux runs, surveille les GPU, conserve le log, extrait les WER de validation, compte les overflows FP16 et vérifie que le checkpoint cible contient tous les états distribués. Son code retour effectif devient non nul si l'entraînement semble réussi mais que le checkpoint est incomplet.

### `src/stage_utils.py`

Gère la reprise entre sessions Kaggle. Il compte les shards, reconnaît un checkpoint complet, retrouve le bon checkpoint sous `/kaggle/input`, crée un lien symbolique en lecture seule dans la nouvelle sortie et détache proprement ce lien après validation du nouveau checkpoint.

### `src/transcribe_dev_panel.py`

Crée une seule fois un panneau déterministe : deux exemples Waxal et deux Klayt du split `dev`, entre 4 et 14 secondes, choisis par SHA-256. Il recharge ensuite ce même panneau à chaque bloc, charge le checkpoint distribué pour l'inférence, transcrit et calcule WER/CER par exemple et en micro-moyenne.

## 13. Description des artefacts de données versionnés

### `baoule_ctc1b_dataset_audit/dataset_audit.csv`

Source de vérité ligne par ligne de l'audit des 2 092 exemples : origine, split, locuteur, texte brut/normalisé, mesures audio, empreintes, statut, actions automatiques et motifs de revue.

### `baoule_ctc1b_dataset_audit/duplicate_groups.json`

Liste les groupes d'empreintes audio ou textuelles répétées et indique si un groupe traverse plusieurs splits. Seuls ces deux fichiers minimaux sont versionnés dans ce dossier afin que le notebook de revue fonctionne dans une nouvelle session.

Les fichiers `dataset_summary.json`, `character_inventory.json` et `manual_review.csv` peuvent être régénérés par le notebook 01. Ils peuvent exister localement comme résultats téléchargés, mais ne sont pas nécessaires au clone minimal versionné.

### `baoule_ctc1b_review/decision_manifest_reviewed.csv`

Manifeste complet de 2 092 lignes avec les décisions humaines effectivement disponibles. C'est l'entrée versionnée de la préparation hackathon.

Les fichiers `decision_manifest.csv`, `review_queue.csv`, `review_queue_summary.json` et `review_progress.json` sont des sorties régénérables de la revue. Ils décrivent respectivement le manifeste de base, la file à écouter, sa composition et la progression humaine.

Les éventuels fichiers suffixés `:Zone.Identifier` sont des métadonnées Windows téléchargées avec les fichiers ; ils ne font pas partie du protocole scientifique et ne sont pas versionnés.

## 14. Description des rapports

- `reports/DATASET_AUDIT_ANALYSIS.md` : analyse détaillée des 2 092 fichiers, durées, fréquences, locuteurs, silences, ratios suspects, doublons, Unicode et décisions recommandées.
- `reports/HACKATHON_DATA_SELECTION.md` : règles exactes et chiffres du sous-ensemble de 1 219 exemples retenu pour le délai du hackathon.
- `reports/SMOKE_TEST_PROTOCOL.md` : objectif, configuration, critères d'acceptation et règles d'interprétation du test de 20 pas.
- `reports/SMOKE_TEST_RESULTS.md` : résultats mesurés, mémoire, temps, checkpoint, WER/UER, overflows et conséquences sur la configuration réelle.
- `reports/.gitkeep` : conserve le dossier dans Git même lorsqu'il ne contient aucun autre fichier ; il n'a pas de rôle expérimental.

## 15. Description des tests

- `tests/test_normalize_text.py` : vérifie apostrophes, lettres/tons, annotations, nombres, NFC et inventaire.
- `tests/test_audit_datasets.py` : vérifie décodage, normalisation et détection des doublons inter-splits.
- `tests/test_create_review_queue.py` : vérifie les priorités, exclusions, segmentation, revue Klayt et champs du manifeste.
- `tests/test_prepare_mixture.py` : vérifie que les décisions humaines gagnent, que Waxal court est retenu, Klayt non revu différé, Waxal long différé et les rognages validés.
- `tests/test_run_smoke_test.py` : vérifie versions optionnelles, agrégation des maxima GPU et intégrité d'un checkpoint distribué.
- `tests/test_run_training_stage.py` : vérifie l'extraction du WER depuis les logs fairseq2 mis en forme.
- `tests/test_stage_utils.py` : vérifie découverte, montage et rejet des checkpoints incomplets.
- `tests/test_transcribe_dev_panel.py` : vérifie le calcul de distance d'édition utilisé par WER/CER.

Lors de la dernière vérification locale du pipeline d'entraînement, 26 tests ont été lancés : 24 ont réussi et 2 ont été ignorés parce que leurs dépendances audio optionnelles n'étaient pas présentes dans l'environnement local. La syntaxe Python, JSON et YAML a aussi été vérifiée.

## 16. Fichiers produits pendant une exécution Kaggle

Ces sorties sont importantes, mais ne doivent généralement pas être ajoutées au dépôt GitHub :

- `baoule_ctc1b_data/` : dataset MixtureParquet matérialisé ; sa copie durable est sur Hugging Face ;
- `smoke_console.log` : journal complet du smoke test ;
- `gpu_metrics.csv` : mesures GPU périodiques ;
- `smoke_summary.json` : synthèse reproductible du smoke test ;
- `stage_XXXXX_YYYYY_console.log` : log d'un bloc réel ;
- `stage_XXXXX_YYYYY_gpu_metrics.csv` : ressources d'un bloc ;
- `stage_XXXXX_YYYYY_summary.json` : métriques, versions et état du checkpoint ;
- `checkpoints/step_N/` : shards trainer/modèle/optimiseur/DataLoader ;
- `dev_panel_manifest.json` : identité du panneau fixe ;
- `dev_panel_predictions_step_N.json` : références, hypothèses, WER et CER du panneau.

Chaque sortie d'entraînement doit être conservée avec **Save Version** avant la fermeture de Kaggle.

## 17. Sécurité, reproductibilité et règles de travail

- Les tokens Hugging Face restent dans le secret Kaggle `HF_TOKEN` et ne sont jamais écrits dans Git.
- Les révisions des deux corpus, du dataset publié et du dépôt Meta sont enregistrées.
- Les dossiers de sortie doivent être vides pour éviter le mélange de deux expériences.
- Le split test n'est pas utilisé pour décider de continuer, arrêter ou choisir un checkpoint.
- Les fichiers longs ne sont jamais découpés arbitrairement avec le texte complet répété sur chaque morceau.
- Une reprise doit retrouver tous les états distribués, pas seulement les poids.
- Les checkpoints incomplets sont rejetés.
- Le panneau de quatre exemples est un indicateur qualitatif stable, pas un remplacement de la validation complète.

## 18. Limites actuelles

1. Le train ne contient que 23 exemples Klayt validés et reste très dominé par Waxal.
2. 1 037 éléments de la file de revue sont encore en attente.
3. Les audios Waxal longs concentrent beaucoup d'heures mais ne sont pas encore segmentés.
4. La limite T4 de 20 secondes filtre 102 exemples du train préparé.
5. Les locuteurs Waxal ne sont pas indépendants entre splits.
6. Le smoke WER au pas 20 est seulement un contrôle technique, pas une performance finale.
7. Le pipeline réel est prêt mais n'a pas encore fourni de checkpoint au pas 250.
8. Les augmentations de bruit et les analyses statistiques finales prévues dans le plan ne sont pas encore intégrées à cette première exécution.
9. Aucun modèle final n'a encore été exporté ou publié.

## 19. Prochaines étapes immédiates

### Étape 1 — Exécuter le premier bloc

Lancer `notebooks/05_train_ctc1b_staged_kaggle.ipynb` dans une session neuve T4 x2 avec Internet et `HF_TOKEN`, en conservant `TARGET_STEP = 250`.

### Étape 2 — Examiner avant de poursuivre

Après `Save Version`, analyser :

- la courbe de loss ;
- les WER de validation aux pas 50, 100, 150, 200 et 250 ;
- le nombre d'overflows ;
- la VRAM, le temps et le disque ;
- les quatre transcriptions du panneau ;
- l'intégrité de tous les shards du checkpoint.

### Étape 3 — Reprendre par blocs

Si le bloc est sain, ajouter sa sortie comme Input d'une nouvelle session, passer `TARGET_STEP` à 500, puis répéter l'analyse. Continuer par incréments de 250 uniquement tant que le `dev` s'améliore de manière utile et que les transcriptions ne se dégradent pas.

### Étape 4 — Sélectionner et évaluer le checkpoint

Choisir le checkpoint sur le split `dev`, puis seulement après ce choix :

- évaluer une fois sur `test` ;
- publier les métriques séparées Waxal/Klayt et la moyenne macro ;
- comparer sur les mêmes audios le modèle de base, le CTC-300M Tree AI Lab et le CTC-1B adapté ;
- analyser les erreurs fréquentes, les tons, les mots rares, les voix et le bruit.

### Étape 5 — Exporter pour Kouman AI

Après validation, convertir ou consolider les shards dans le format d'inférence attendu, tester le rechargement, publier le modèle et sa fiche sur le compte Hugging Face Tree AI Lab, puis mesurer la latence et la mémoire dans le pipeline oral complet.

## 20. Suite après le hackathon

Le travail de fond devra continuer :

- terminer la revue des Klayt ;
- corriger les transcriptions dont seule une partie correspond à l'audio ;
- aligner et segmenter les longs Waxal en couples audio–texte fiables ;
- récupérer les 820 exemples différés quand leur qualité est suffisante ;
- enrichir les voix et conditions acoustiques ;
- tester des augmentations audio contrôlées ;
- vérifier complètement la couverture du tokenizer ;
- construire un test indépendant supplémentaire ;
- comparer éventuellement plusieurs stratégies de mélange ;
- expérimenter `omniASR_LLM_300M` comme prévu ;
- itérer avec les erreurs réellement observées dans Kouman AI.

## 21. Conclusion

Le projet a franchi toutes les étapes préparatoires indispensables : choix du CTC-1B après benchmark, audit des données, normalisation, revue partielle, sélection hackathon, création et publication du dataset, validation du DataLoader et smoke test distribué avec checkpoint complet.

La principale valeur du nouveau protocole tient autant à la qualité des données et à la reproductibilité qu'à la taille du modèle. Le prochain résultat décisif sera le premier bloc réel de 250 pas. Il permettra de déterminer, à partir de mesures et de transcriptions concrètes, si l'entraînement doit continuer, être ajusté ou s'arrêter sur un checkpoint déjà satisfaisant.
