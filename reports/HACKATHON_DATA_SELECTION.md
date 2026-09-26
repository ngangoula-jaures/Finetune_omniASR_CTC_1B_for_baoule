# Sélection des données pour le hackathon

## Objectif

Cette sélection constitue un sous-ensemble immédiatement exploitable pour le
premier fine-tuning de `omniASR_CTC_1B`. Elle ne remplace pas la revue complète
prévue après le hackathon.

La source de vérité est le manifeste
`baoule_ctc1b_review/decision_manifest_reviewed.csv` contenant 2 092 exemples.

## Règles appliquées

Les décisions humaines sont prioritaires :

- `exclude` : exemple exclu, y compris s'il s'agit d'un Waxal court ;
- `segment` : exemple différé jusqu'à la segmentation alignée ;
- `trim` : exemple retenu uniquement si les bornes sont valides et si la durée
  finale ne dépasse pas 40 secondes ;
- `keep` : exemple retenu si sa durée ne dépasse pas 40 secondes.

Pour les exemples sans décision finale :

- les Waxal non rejetés de 40 secondes maximum sont retenus ;
- les Waxal supérieurs à 40 secondes sont différés ;
- les Klayt encore non révisés sont différés.

Les splits d'origine sont conservés. `validation` devient simplement `dev` dans
le partitionnement attendu par OmniASR.

## Résultat

| Source | Split de sortie | Exemples | Durée approximative |
|---|---:|---:|---:|
| Klayt | train | 23 | 0,061 h |
| Klayt | dev | 15 | 0,034 h |
| Klayt | test | 212 | 0,440 h |
| Waxal | train | 783 | 2,556 h |
| Waxal | dev | 95 | 0,326 h |
| Waxal | test | 91 | 0,288 h |
| **Total** | — | **1 219** | **3,706 h** |

Le split d'entraînement contient 806 exemples, soit environ 2,617 heures.

Sur les 2 092 exemples initiaux :

- 1 219 sont sélectionnés ;
- 820 sont différés ;
- 53 sont exclus manuellement.

Les 820 exemples différés comprennent 595 Klayt encore non révisés, 68 audios
explicitement marqués pour segmentation et 157 autres Waxal de plus de
40 secondes.

## Format produit

Les audios sélectionnés seront décodés, convertis en mono 16 kHz puis encodés
en FLAC. Les Parquet utiliseront des groupes de lignes de 100 et le
partitionnement suivant :

```text
baoule_mixed/version=0/
├── corpus=waxal/
│   ├── split=train/
│   ├── split=dev/
│   └── split=test/
└── corpus=klayt/
    ├── split=train/
    ├── split=dev/
    └── split=test/
```

Toutes les partitions utilisent `language=bci_Latn`.

Le dépôt Hugging Face prévu pour cette version est
`Tree-AI-lab/baoule-asr-hackathon-mixture`. Il est créé en privé par défaut ;
sa visibilité pourra être changée explicitement après vérification des fichiers
et de la fiche de dataset.

## Limites connues

- Le train Klayt validé est encore petit : 23 exemples.
- Le train est donc très majoritairement composé de Waxal.
- Les locuteurs Waxal ne sont pas disjoints entre les splits d'origine.
- Les données différées ne seront récupérées qu'après revue ou segmentation.
- Cette version vise le délai du hackathon, pas la constitution définitive du
  futur dataset Tree AI Lab.
