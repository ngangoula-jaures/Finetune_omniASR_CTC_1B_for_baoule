# Analyse de l'audit WaxalNLP + Baoulé Common Voice

## 1. Portée

Cette analyse repose sur l'audit Kaggle des révisions suivantes :

- `google/WaxalNLP`, configuration `bau_tts`, révision `5f4d8ca24f2b9d168b2ee545f1febaaff4b40580` ;
- `Klayt/baoule-common-voice`, révision `96eb4413b7cc3190c39a35ee03eccb321c2b9929`.

Les trois splits de chaque source ont été parcourus. Les 2 092 fichiers ont tous été décodés. Aucun fichier vide, silencieux ou illisible n'a été rejeté automatiquement et aucun doublon audio exact n'a été trouvé.

Après correction de l'audit pour séparer les transformations automatiques des défauts à examiner :

- 1 480 exemples sont acceptés directement ;
- 612 exemples sont marqués pour revue ;
- aucun exemple n'est rejeté automatiquement ;
- les 1 216 fichiers Waxal portent l'action automatique `resample_to_16khz`, sans être signalés pour cette seule raison.

## 2. Vue d'ensemble

| Source | Split | Exemples | Locuteurs | Durée totale | Exemples ≤ 35 s | Durée ≤ 35 s | Exemples > 35 s | Durée > 35 s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Klayt | train | 319 | 3 | 0,694 h | 319 | 0,694 h | 0 | 0 h |
| Klayt | validation | 267 | 3 | 0,506 h | 267 | 0,506 h | 0 | 0 h |
| Klayt | test | 290 | 9 | 0,599 h | 290 | 0,599 h | 0 | 0 h |
| Waxal | train | 972 | 4 | 10,975 h | 774 | 2,457 h | 198 | 8,518 h |
| Waxal | validation | 122 | 4 | 1,483 h | 91 | 0,284 h | 31 | 1,199 h |
| Waxal | test | 122 | 4 | 1,596 h | 89 | 0,259 h | 33 | 1,338 h |

Le volume total audité est d'environ 15,854 heures : 14,055 heures de Waxal et 1,799 heure de Klayt.

Sans segmenter les fichiers de plus de 35 secondes, le train directement exploitable contient au maximum :

- 774 fichiers Waxal, soit 2,457 heures ;
- 319 fichiers Klayt, soit 0,694 heure ;
- 1 093 fichiers et environ 3,151 heures au total avant les autres contrôles de qualité.

La segmentation alignée des fichiers longs Waxal peut potentiellement récupérer jusqu'à 8,518 heures supplémentaires pour l'entraînement. Elle constitue donc une étape importante, mais elle ne doit pas être faite par découpage arbitraire.

Avec les règles automatiques strictes, 836 exemples du train sont acceptés sans signalement, pour environ 2,292 heures :

- Waxal : 547 exemples et 1,681 heure ;
- Klayt : 289 exemples et 0,611 heure.

Si les silences initiaux et finaux sont traités comme une transformation de rognage plutôt que comme un défaut bloquant, le premier mélange court peut atteindre 1 089 exemples et environ 3,141 heures :

- Waxal : 770 exemples et 2,447 heures ;
- Klayt : 319 exemples et 0,694 heure.

Ces chiffres n'autorisent pas encore l'entraînement définitif : les fichiers Klayt doivent être contrôlés pour la présence de voix non transcrites.

## 3. Fréquences d'échantillonnage

| Source | Fréquence originale | Canaux |
|---|---:|---:|
| Waxal | 48 kHz pour 1 216/1 216 fichiers | mono |
| Klayt | 16 kHz pour 876/876 fichiers | mono |

Waxal devra être rééchantillonné explicitement de 48 kHz vers 16 kHz pendant la préparation. Il s'agit d'une opération normale et automatique, pas d'un défaut nécessitant une écoute manuelle.

Le notebook CTC-300M précédent effectuait déjà ce rééchantillonnage au moyen de `cast_column("audio", Audio(sampling_rate=16_000))`. L'audit lit volontairement les fichiers originaux avec `decode=False`, ce qui explique pourquoi il observe 48 kHz.

## 4. Durées et textes multilignes de Waxal

Waxal contient 262 fichiers de plus de 35 secondes, représentant environ 11,055 heures. Parmi les 1 216 exemples Waxal :

- 251 transcriptions brutes contiennent plusieurs lignes ;
- 183 fichiers sont à la fois multilignes et supérieurs à 35 secondes ;
- 79 fichiers longs ont une transcription sur une seule ligne ;
- 68 fichiers courts ont une transcription multiligne.

Les fichiers longs ne peuvent pas être répartis en segments audio sans répartir aussi leur transcription. Le futur pipeline devra utiliser un alignement CTC ou une annotation manuelle pour obtenir des couples segment–texte fiables.

## 5. Locuteurs et indépendance des splits

### 5.1 Klayt Common Voice

Les identifiants de locuteurs sont disjoints :

- 3 locuteurs dans `train` ;
- 3 autres locuteurs dans `validation` ;
- 9 autres locuteurs dans `test` ;
- aucune intersection de `client_id` entre les trois splits.

Cette organisation permet une évaluation réelle sur des voix absentes de l'entraînement.

### 5.2 Waxal

Les mêmes quatre locuteurs `JH`, `JK`, `KK` et `RK` apparaissent dans `train`, `validation` et `test`. Les métriques Waxal évaluent donc de nouveaux énoncés, mais pas une généralisation à de nouveaux locuteurs.

Les résultats finaux devront présenter séparément :

- Waxal, pour la continuité avec l'ancien modèle ;
- Klayt, pour la généralisation à des locuteurs non vus ;
- la moyenne macro des deux sources.

## 6. Silences et niveau sonore

### 6.1 Klayt

Soixante-dix fichiers ont été signalés :

- 29 avec un long silence initial uniquement ;
- 20 avec un long silence final uniquement ;
- 4 avec les deux ;
- 17 avec un niveau RMS très faible.

Le split d'entraînement compte 30 fichiers signalés, la validation 15 et le test 25.

Un locuteur du test Klayt, dont l'identifiant commence par `ac0f60ef`, représente 13 fichiers et ses 13 fichiers sont à faible niveau. Ce sous-ensemble peut servir de test de robustesse au faible volume. Il ne doit pas être supprimé automatiquement du test.

Les fichiers de train contenant seulement du silence avant ou après la phrase pourront être rognés avec une marge. Ceux qui contiennent une autre voix doivent être écoutés et découpés autour de la parole cible, ou exclus si l'alignement reste ambigu.

Après le nouvel audit, Klayt contient 806 exemples acceptés directement et 70 en revue. Le train contient 289 acceptés et 30 signalés uniquement pour leurs silences. Comme l'utilisateur a observé des voix précédant parfois le traducteur, l'absence de signalement automatique ne garantit pas l'absence de parole parasite : le détecteur mesure les marges silencieuses, il ne réalise pas de diarisation.

### 6.2 Waxal

Parmi les fichiers Waxal de 35 secondes ou moins :

- 266 présentent un silence initial supérieur au seuil de 1,5 seconde ;
- 15 présentent un silence final supérieur au seuil de 2 secondes.

Ces signaux seront revus par échantillonnage et pourront être rognés automatiquement si le début de la parole cible est conservé.

## 7. Ratios texte–audio suspects

Le ratio médian est proche de 1,37 mot par seconde dans les deux sources. Un seuil conservateur de revue, inférieur à 0,4 ou supérieur à 3,5 mots par seconde, signale onze exemples :

| Source | Split | Identifiant | Mots/s | Durée | Mots |
|---|---|---|---:|---:|---:|
| Waxal | train | `bau_JK_JK_BCI_2_107` | 17,404 | 13,0 s | 227 |
| Waxal | train | `bau_JK_JK_BCI_5_241` | 0,166 | 735,6 s | 122 |
| Waxal | train | `bau_KK_KK_BCI_1_31c` | 0,355 | 8,4 s | 3 |
| Waxal | train | `bau_RK_RK_BCI_3_219` | 5,204 | 252,7 s | 1 315 |
| Waxal | train | `bau_KK_KK_BCI_3_211` | 3,721 | 408,5 s | 1 520 |
| Waxal | train | `bau_JH_JH_BCI_2_126` | 5,514 | 12,0 s | 66 |
| Waxal | train | `bau_JH_JH_BCI_2_138` | 3,591 | 3,6 s | 13 |
| Waxal | validation | `bau_JK_JK_BCI_4_236` | 3,594 | 88,5 s | 318 |
| Waxal | test | `bau_JK_JK_BCI_1_49` | 11,555 | 8,0 s | 92 |
| Waxal | test | `bau_JK_JK_BCI_4_231` | 7,415 | 377,6 s | 2 800 |
| Klayt | test | `common_voice_bci_41488812.mp3` | 4,575 | 3,1 s | 14 |

Les cas les plus extrêmes sont incompatibles avec une lecture normale et indiquent probablement une transcription comprenant du texte absent de l'audio, un fichier incomplet ou un regroupement incorrect. Ils doivent être écoutés. Ils seront exclus de l'entraînement ou de l'évaluation propre tant que leur alignement n'est pas confirmé.

## 8. Doublons

### 8.1 Audio

Aucun doublon audio exact n'a été trouvé parmi les 2 092 fichiers.

### 8.2 Texte

Cinq groupes de transcriptions normalisées identiques ont été trouvés :

- quatre groupes uniquement dans le train Waxal ;
- un groupe traversant `train` et `test` Waxal.

Le groupe inter-split concerne :

- train : `bau_KK_KK_BCI_4_227` ;
- test : `bau_KK_KK_BCI_4_239`.

Les audios ne sont pas identiques, mais la longue transcription normalisée est identique. Pour conserver l'indépendance textuelle du test, l'exemple d'entraînement `bau_KK_KK_BCI_4_227` devra être retiré du train avant toute segmentation.

Les quatre doublons internes au train sont des répétitions par le même locuteur avec des durées différentes. Ils seront écoutés afin de conserver la meilleure prise ou de vérifier qu'ils apportent une variation acoustique utile sans surpondérer la phrase.

## 9. Inventaire Unicode

Après normalisation :

- Waxal contient 61 caractères non blancs distincts ;
- Klayt en contient 44 ;
- les lettres baoulé centrales `ɛ`, `ɔ` et `ɲ`, les apostrophes et les marques de ton sont conservées ;
- Klayt contient notamment `à` et `ĩ`, absents de Waxal ;
- Waxal contient plusieurs caractères provenant de mots français ou de noms propres, ainsi que quelques chiffres intégrés à des tokens.

Les deux corpus emploient des variantes d'apostrophes dans leur forme brute. La normalisation vers `'` est justifiée pour éviter plusieurs représentations typographiques du même phénomène. Les tons ne doivent pas être supprimés.

L'étape suivante devra encore vérifier l'encodage de chaque caractère normalisé avec `omniASR_tokenizer_v1`.

## 10. Décision de préparation

### 10.1 Utilisation immédiate possible après transformations automatiques

- rééchantillonnage Waxal 48→16 kHz ;
- conservation de Klayt à 16 kHz ;
- conversion mono déterministe ;
- normalisation Unicode et des apostrophes ;
- suppression du doublon train–test identifié ;
- exclusion provisoire des ratios texte–audio extrêmes ;
- limite à 35 secondes pour le premier smoke test.

### 10.2 Revue humaine nécessaire

- les 70 fichiers Klayt signalés pour silence ou faible RMS ;
- les onze ratios texte–audio extrêmes ;
- un échantillon des fichiers Waxal à long silence initial ;
- les quatre groupes de doublons textuels internes au train ;
- les fichiers Klayt dans lesquels une voix non transcrite précède le locuteur cible.

### 10.3 Travail spécifique avant récupération des audios longs

- alignement CTC des fichiers Waxal de plus de 35 secondes ;
- découpage aux frontières validées ;
- association d'un texte différent à chaque segment ;
- rejet des segments de faible confiance ;
- nouvelle vérification des doublons après segmentation.

## 11. Prochaine étape

La prochaine phase d'exécution doit produire :

1. un notebook d'écoute des exemples signalés ;
2. un manifeste de décisions `keep`, `trim`, `segment`, `exclude` ;
3. un contrôle de couverture du tokenizer ;
4. un premier MixtureParquet limité aux exemples courts validés ;
5. un test du dataloader Meta avant tout chargement du CTC-1B.

Le nouvel audit contient maintenant 612 lignes de revue ciblées au lieu des 1 286 lignes initiales. La prochaine sélection devra distinguer :

- les silences rognables automatiquement ;
- les fichiers longs à segmenter ;
- les ratios texte–audio à vérifier ;
- la fuite textuelle train–test à supprimer ;
- les voix parasites de Klayt qui nécessitent une écoute.
