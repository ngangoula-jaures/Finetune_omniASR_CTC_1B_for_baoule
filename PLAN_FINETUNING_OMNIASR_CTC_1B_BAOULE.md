# Plan complet de fine-tuning d'OmniASR CTC-1B pour le baoulé

## 1. Objet du document

Ce document décrit le protocole proposé pour adapter `omniASR_CTC_1B` au baoulé à partir d'un mélange de deux corpus :

- `google/WaxalNLP`, configuration `bau_tts` ;
- `Klayt/baoule-common-voice`.

L'objectif expérimental est d'obtenir un modèle plus précis et plus robuste que le checkpoint actuel `Tree-AI-lab/omniASR-CTC-300M-baoule-1000steps`, tout en produisant des résultats reproductibles et évalués sur des données réellement indépendantes.

Une amélioration n'est pas considérée comme acquise du seul fait que le nouveau modèle est plus grand. Elle devra être démontrée sur des jeux de test gelés, avec des intervalles de confiance et une analyse séparée par source, locuteur et condition acoustique.

Ce dossier ne contient pour l'instant que le plan. Le script ou notebook de préparation et d'entraînement sera écrit après validation de ce document.

## 2. Décisions déjà prises et points à verrouiller

### 2.1 Checkpoint de départ

Le checkpoint prévu est :

```text
omniASR_CTC_1B
```

Il s'agit du modèle évalué dans le benchmark local. Le dépôt officiel Meta propose désormais aussi des checkpoints suffixés `_v2`, annoncés comme améliorés, mais ceux-ci n'ont pas été inclus dans ce benchmark. Le script ne changera donc pas silencieusement de checkpoint. La révision exacte du dépôt OmniASR et l'identifiant exact du modèle devront être figés dans la configuration de l'expérience.

### 2.2 Tokenizer

La première série d'expériences conservera `omniASR_tokenizer_v1` et la tête CTC existante. Remplacer immédiatement le tokenizer obligerait à réinitialiser au moins la projection de sortie et rendrait plus difficile l'exploitation des connaissances déjà présentes dans le checkpoint.

Avant l'entraînement, un audit mesurera néanmoins :

- la couverture de tous les caractères baoulé ;
- la présence de tokens inconnus ;
- le coût en tokens des transcriptions ;
- les différences entre apostrophes, diacritiques et variantes Unicode ;
- la capacité du tokenizer à représenter les graphies des deux corpus sans perte d'information.

Un tokenizer spécialisé ne sera étudié que dans une expérience séparée si cet audit révèle un problème réel de couverture ou de segmentation.

## 3. Références techniques et état des données

La [recette CTC officielle d'Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr/tree/main/workflows/recipes/wav2vec2/asr) permet de fine-tuner un checkpoint CTC existant avec le backend `MixtureParquetStorage`. La configuration officielle recommandée utilise notamment un learning rate de `1e-5`, une durée maximale de 60 secondes et un maximum de 5 000 pas comme point de départ pour les langues peu dotées.

La [configuration officielle `ctc-finetune-recommendation.yaml`](https://github.com/facebookresearch/omnilingual-asr/blob/main/workflows/recipes/wav2vec2/asr/configs/ctc-finetune-recommendation.yaml) sera copiée à une révision figée puis adaptée explicitement, plutôt que reconstruite à partir de valeurs implicites susceptibles de changer.

Le dépôt [Klayt/baoule-common-voice](https://huggingface.co/datasets/Klayt/baoule-common-voice) contient actuellement 876 exemples : environ 319 en entraînement, 267 en validation et 290 en test. Sa fiche indique de l'audio mono 16 kHz, les colonnes `audio`, `sentence`, `client_id`, `path` et `locale`, ainsi que des caractères baoulé comme `ɛ`, `ɔ`, `ŋ`, `ʼ` et des voyelles portant des tons. Le visualiseur indique seulement trois valeurs de `client_id` dans le split d'entraînement ; cette faible diversité de locuteurs devra être prise en compte dans l'interprétation des résultats.

La configuration baoulé de [google/WaxalNLP](https://huggingface.co/datasets/google/WaxalNLP) appartient à la partie TTS de Waxal. La fiche du corpus décrit ces données comme des lectures de haute qualité et indique une licence `CC-BY-4.0` pour les données fournies par l'Université du Ghana, dont le baoulé.

Les deux datasets sont distants et peuvent évoluer. Chaque exécution enregistrera donc :

- l'identifiant du dataset ;
- sa configuration ;
- sa révision Hugging Face exacte ;
- les empreintes des splits ;
- le nombre d'exemples et la durée avant et après filtrage ;
- la licence affichée à la révision utilisée.

## 4. Limites identifiées dans l'entraînement CTC-300M précédent

Le premier entraînement a permis d'améliorer fortement le CTC-300M, mais son protocole comporte des limites qui seront corrigées pour le CTC-1B.

| Limite observée | Conséquence possible | Correction prévue |
|---|---|---|
| Utilisation de Waxal uniquement | Faible diversité acoustique et peu de robustesse hors domaine | Mélange contrôlé Waxal + Common Voice |
| Maximum de 1 000 pas | Convergence potentiellement incomplète | Jusqu'à 5 000 pas, avec arrêt anticipé fondé sur la validation |
| `max_num_elements` limité à l'équivalent d'un seul audio de 60 s | Gradient plus bruité et faible volume audio par microbatch | Batching dynamique selon la mémoire et accumulation visant une durée audio effective stable |
| Absence d'augmentation audio explicite | Généralisation limitée au bruit, aux codecs et aux microphones | Augmentations contrôlées et mesurées |
| Aucun audit systématique des doublons ou fuites de locuteurs | Résultats potentiellement optimistes | Déduplication et vérification des intersections entre splits |
| Normalisation de `’` et `‘`, mais pas de `ʼ` | Orthographes différentes entre les deux corpus | Canonicalisation explicite et testée des apostrophes |
| Fréquence supposée égale à 16 kHz lors du calcul de durée | Durée ou signal incorrect si un exemple ne respecte pas cette hypothèse | Lecture du taux réel, assertion puis rééchantillonnage explicite |
| Rejet des audios longs sans tentative d'alignement | Perte d'une part importante des données | Segmentation seulement après alignement fiable, sinon exclusion documentée |
| Évaluation finale sur les dix premiers exemples compatibles | Échantillon trop petit et non représentatif | Évaluation exhaustive sur les tests gelés |
| Sélection et benchmark sur le même split de validation Waxal | Estimation optimiste du checkpoint sélectionné | Séparation stricte entre développement et test final |
| Sauvegarde `save_model_only` | Reprise exacte impossible après interruption | Sauvegarde périodique de l'état complet du trainer et export séparé du modèle |
| Sélection sur un WER global unique | Le corpus majoritaire peut masquer une régression sur l'autre domaine | Métriques séparées par corpus et score macro de sélection |
| Entraînement jusqu'à 60 s alors que l'inférence standard est limitée à moins de 40 s | Décalage entre entraînement et déploiement | Segments alignés d'au plus 30–35 s pour la recette principale |

Le passage de 300M à 1B augmente la capacité, mais aussi les risques de surapprentissage, d'instabilité et de saturation mémoire. Les corrections de données et d'évaluation sont donc aussi importantes que le changement de taille du modèle.

## 5. Principe fondamental : qualité de l'alignement audio–texte

Le CTC suppose que la transcription correspond à la parole contenue dans le signal. Il tolère du silence et du bruit acoustique, mais pas une quantité importante de parole absente de la transcription.

Les cas observés ou anticipés dans Common Voice seront séparés en trois catégories :

1. **Bruit non vocal** : ventilation, circulation, souffle de microphone ou bruit de pièce. L'exemple peut être conservé si la parole cible reste intelligible.
2. **Silence ou bruit avant/après la phrase** : l'exemple peut être rogné avec une marge afin de conserver les consonnes initiales et finales.
3. **Voix d'une autre personne, consigne orale ou conversation non transcrite** : cette portion doit être découpée de façon fiable. Si elle ne peut pas être séparée sans ambiguïté, l'exemple doit être rejeté ou corrigé manuellement.

Conserver de la parole parasite comme si elle était un simple bruit apprendrait au modèle à ignorer des mots réellement prononcés et rendrait l'alignement CTC contradictoire.

## 6. Pipeline d'audit des données

L'audit sera exécuté avant toute conversion au format Meta. Il produira un fichier CSV ou Parquet par split et un résumé JSON.

### 6.1 Contrôles audio automatiques

Pour chaque fichier :

- décodage réussi ;
- nombre de canaux ;
- fréquence d'échantillonnage réelle ;
- durée ;
- pic, RMS et éventuel écrêtage ;
- proportion de silence ;
- durée de silence initial et final ;
- présence estimée de plusieurs zones de parole ;
- rapport entre durée audio et longueur de la transcription ;
- empreinte du signal pour détecter les doublons exacts ;
- empreinte perceptuelle ou représentation acoustique pour rechercher les quasi-doublons.

Tous les signaux acceptés seront convertis en mono 16 kHz avec une méthode de rééchantillonnage déterministe. La normalisation d'amplitude du dataloader sera conservée, mais aucun écrasement dynamique destructif ne sera appliqué aux fichiers sources.

### 6.2 Contrôles textuels

Pour chaque transcription :

- normalisation Unicode NFC ;
- suppression seulement des annotations identifiées comme non prononcées ;
- canonicalisation des espaces ;
- conversion contrôlée de `’`, `‘`, `ʼ` et caractères équivalents vers une apostrophe canonique ;
- inventaire des lettres, marques combinantes, chiffres et symboles ;
- détection des textes vides, anormalement courts ou anormalement longs ;
- encodage puis décodage avec le tokenizer ;
- rejet des exemples contenant un token inconnu ou une perte de caractères ;
- recherche des transcriptions dupliquées dans un même split et entre les splits.

Les accents et tons ne seront pas supprimés automatiquement. Une différence réellement linguistique ne doit pas être effacée au nom de la normalisation. La politique finale sera documentée avec des exemples et, idéalement, vérifiée par une personne maîtrisant l'orthographe baoulé.

### 6.3 Écoute humaine stratifiée

Un échantillon sera écouté avant l'entraînement :

- exemples tirés de chaque source et de chaque locuteur ;
- fichiers les plus bruyants ;
- fichiers présentant le plus de silence ;
- rapports texte/durée atypiques ;
- doublons potentiels ;
- fichiers détectés comme multi-voix ;
- fichiers proches des seuils de rejet.

L'audit humain produira des étiquettes simples : `clean`, `noisy_aligned`, `trim`, `mismatch`, `multi_speaker` et `reject`.

## 7. Construction des splits

### 7.1 Règles de non-fuite

Les identifiants de locuteurs seront préfixés par la source, par exemple `waxal:JH` ou `klayt:<client_id>`, afin d'éviter toute collision artificielle.

Le pipeline vérifiera les intersections suivantes :

- locuteurs train/dev/test ;
- chemins et identifiants audio ;
- empreintes audio exactes ou proches ;
- transcriptions identiques ou presque identiques.

Les splits `test` publiés ne seront jamais ajoutés à l'entraînement. Si les splits publiés contiennent les mêmes locuteurs, ils seront conservés pour la comparabilité, mais une évaluation supplémentaire réellement indépendante par locuteur sera créée.

### 7.2 Deux niveaux d'évaluation

Le protocole distinguera :

1. **Évaluation officielle par source** : splits `test` fournis par Waxal et Klayt, après les seules corrections nécessaires au décodage et à la normalisation.
2. **Évaluation de généralisation** : sous-ensemble gelé, dédupliqué et aussi indépendant que possible en locuteurs et conditions acoustiques.

Avec seulement quelques locuteurs, une séparation parfaite peut réduire fortement les données. Si nécessaire, une validation croisée par locuteur sera utilisée pour analyser la stabilité, tandis qu'un test final restera verrouillé pour la comparaison des checkpoints définitifs.

### 7.3 Conservation des domaines

Les exemples garderont une colonne `corpus` (`waxal` ou `klayt`) et des métadonnées de qualité. Cela permettra de calculer les métriques séparément et d'éviter qu'un corpus plus volumineux masque les erreurs de l'autre.

## 8. Traitement des fichiers longs

Les fichiers ne seront jamais découpés arbitrairement avec une transcription complète répétée sur chaque segment.

Pour un fichier dépassant la durée cible :

1. produire une première transcription et un alignement avec le modèle de base ou un aligneur CTC ;
2. estimer les frontières temporelles des tokens ou des mots ;
3. découper aux silences proches de ces frontières ;
4. limiter les segments à environ 30–35 secondes ;
5. associer à chaque segment uniquement le texte aligné ;
6. rejeter ou envoyer en révision manuelle les alignements de faible confiance.

En l'absence d'alignement fiable, le fichier sera exclu et comptabilisé dans le rapport de préparation. Une donnée longue mal alignée est plus nuisible qu'une donnée non utilisée.

## 9. Mélange de Waxal et Common Voice

Une simple concaténation n'est pas suffisante, car elle donne davantage de poids au corpus qui contient le plus d'heures ou les segments les plus longs.

Le format MixtureParquet comportera au minimum deux corpus distincts :

```text
corpus=waxal
corpus=klayt
```

Le premier ratio testé sera fondé sur la durée audio acceptée après audit, avec une limite empêchant le petit corpus d'être soit écrasé, soit répété excessivement. Les ratios exacts ne seront fixés qu'après calcul des heures propres et bruyantes. Une petite ablation comparera au moins :

- échantillonnage proportionnel à la durée ;
- mélange équilibré par source ;
- mélange intermédiaire donnant davantage de poids à Waxal propre tout en exposant régulièrement le modèle à Klayt.

Le choix sera effectué sur la moyenne macro des validations Waxal et Klayt, et non sur la seule métrique concaténée.

### 9.1 Curriculum éventuel

Une expérience pourra commencer par des données propres et bien alignées, puis introduire progressivement les exemples `noisy_aligned`. Ce curriculum ne sera conservé que s'il améliore les validations propres et bruitées. Les exemples mal alignés ne feront partie d'aucune étape.

## 10. Augmentations audio

Les augmentations seront réalisées uniquement sur le split d'entraînement et de manière aléatoire, sans créer de copies dans les splits de validation ou de test.

### 10.1 Augmentations proposées

- légère variation de vitesse, par exemple autour de 0,9×, 1,0× et 1,1× ;
- variation de gain sans écrêtage ;
- ajout de bruit de fond non vocal à plusieurs niveaux de SNR ;
- réverbération légère à partir de réponses impulsionnelles autorisées ;
- simulation de bande téléphonique ou de codecs compressés ;
- SpecAugment ou masquage temps/fréquence si la recette et l'architecture le gèrent correctement.

### 10.2 Règles de sécurité des augmentations

- Ne pas modifier le signal au point de rendre la transcription fausse.
- Ne pas utiliser une autre voix intelligible comme bruit sans politique explicite de parole concurrente.
- Ne pas appliquer toutes les transformations à chaque exemple.
- Journaliser les probabilités et plages de paramètres.
- Mesurer séparément l'effet de l'augmentation au moyen d'une ablation.

Le bruit réel de Klayt ne remplace pas un contrôle de qualité : seul le bruit compatible avec la transcription est utile.

## 11. Configuration d'entraînement proposée

Les valeurs ci-dessous constituent le point de départ, pas une garantie d'optimalité.

| Paramètre | Valeur ou stratégie initiale |
|---|---|
| Modèle | `omniASR_CTC_1B` à révision figée |
| Tokenizer | `omniASR_tokenizer_v1` |
| Fréquence audio | mono 16 kHz |
| Durée minimale | 1 à 2 s, fixée après audit |
| Durée maximale principale | 30–35 s après segmentation alignée |
| Optimiseur | celui de la recette CTC officielle, état intégral sauvegardé |
| Learning rate | `1e-5` comme référence ; comparaison limitée avec `5e-6` si nécessaire |
| Précision | BF16 sur GPU compatible ; FP16 contrôlé sur T4 |
| Activation checkpointing | couche par couche |
| Gradient clipping | norme maximale 1,0 |
| Nombre maximal de pas | 5 000 |
| Validation | toutes les 500 étapes par défaut |
| Checkpoints | toutes les 500 étapes, état complet |
| Transcription témoin | après chaque checkpoint de 500 étapes |
| Arrêt anticipé | patience définie sur plusieurs validations, après une durée minimale d'entraînement |
| Seeds | au moins trois pour l'expérience finale si le budget le permet |

### 11.1 Batching par durée

Le batch sera limité par le nombre total d'échantillons audio plutôt que par un nombre fixe de fichiers. L'objectif est une durée audio effective stable par mise à jour, obtenue par :

- un microbatch adapté à la mémoire réelle ;
- une accumulation de gradients plus élevée lorsque le microbatch est petit ;
- le regroupement approximatif des durées pour réduire le padding ;
- un mélange aléatoire suffisant entre les sources et les locuteurs.

La taille effective sera validée avec un test mémoire avant le run complet. Les valeurs ne doivent pas être copiées directement du 300M, car la mémoire du 1B est très différente.

### 11.2 Gel temporaire ou fine-tuning complet

La recette officielle recommandée démarre avec l'encodeur non gelé. Ce sera la référence principale. Une variante courte pourra comparer un gel initial de quelques centaines de pas, pendant lequel seule la tête CTC est ajustée, suivi d'un dégel complet.

Cette variante ne sera retenue que si elle réduit l'instabilité ou améliore la validation. Geler durablement l'encodeur limiterait l'adaptation acoustique recherchée.

### 11.3 Scheduler et warm-up

Le scheduler effectif et son warm-up seront écrits explicitement dans le fichier de configuration ou enregistrés depuis les valeurs par défaut de fairseq2. Aucun paramètre important ne devra dépendre d'une valeur implicite non archivée.

### 11.4 Transcription témoin toutes les 500 étapes

Un audio témoin fixe sera choisi dans le split de validation, jamais dans le test final. Il devra être assez court pour être transcrit rapidement et représentatif d'une difficulté utile, sans être un cas manifestement corrompu.

Après les pas 500, 1 000, 1 500, etc., le notebook devra :

1. terminer et sauvegarder le checkpoint courant ;
2. recharger ou utiliser ce checkpoint en mode évaluation ;
3. transcrire le même audio témoin avec le décodage greedy ;
4. afficher la référence, la prédiction et le WER/CER de cet exemple ;
5. enregistrer la prédiction dans un historique CSV ou JSONL ;
6. afficher en parallèle les métriques complètes des validations Waxal et Klayt ;
7. libérer les objets d'inférence et le cache GPU avant de reprendre l'entraînement.

L'historique rendra visible l'évolution qualitative :

```text
step,reference,prediction,wer,cer,checkpoint
500,...
1000,...
1500,...
```

Le même exemple peut être observé à chaque étape pour suivre les changements, mais il devient alors un exemple de développement. Il ne doit pas être utilisé comme preuve finale de qualité. Une transcription qui paraît bonne peut masquer une dégradation sur les autres phrases ; la décision d'arrêter doit donc combiner l'écoute de cet exemple et les métriques complètes de validation.

Pour éviter de tirer des conclusions d'un seul type de voix, le notebook pourra aussi afficher un petit panneau secondaire de trois à cinq exemples fixes, sans rendre leur transcription obligatoire pour poursuivre le run. L'audio témoin principal restera toujours affiché.

## 12. Contraintes mémoire du modèle 1B

Le modèle compte environ 975 millions de paramètres. En entraînement Adam en précision mixte, les poids, gradients, poids maîtres et moments de l'optimiseur peuvent à eux seuls approcher ou dépasser la mémoire d'une T4 de 16 Gio, avant même les activations.

### 12.1 Environnement privilégié

- GPU de 40 Gio ou davantage pour un fine-tuning complet plus simple ;
- BF16 lorsque le matériel le supporte ;
- entraînement distribué ou sharding si plusieurs GPU sont disponibles ;
- version de PyTorch, CUDA, fairseq2 et `omnilingual-asr` entièrement figée.

### 12.2 Mode contraint

Sur une T4 ou un GPU de 16 Gio, un fine-tuning complet peut être impossible sans mécanisme supplémentaire. Les options seront évaluées dans cet ordre :

1. FP16 et activation checkpointing ;
2. réduction de la durée maximale du microbatch ;
3. accumulation de gradients ;
4. sharding FSDP ou offload compatible avec la recette ;
5. fine-tuning paramètre-efficace seulement si son implémentation est stable et vérifiée.

Une pull request LoRA existe dans le dépôt OmniASR, mais elle ne doit pas être considérée comme une fonctionnalité officielle stable tant qu'elle n'est pas intégrée. Le premier script ne dépendra donc pas d'une modification non publiée de la recette, sauf décision explicite après un test séparé.

### 12.3 Adaptation aux ressources Kaggle disponibles

La session Kaggle montrée dispose de :

- deux GPU NVIDIA T4 d'environ 15 Gio chacun ;
- environ 30 Gio de RAM système ;
- environ 57 Gio de disque au maximum ;
- une durée maximale de session de 12 heures.

Les mémoires des deux T4 ne forment pas automatiquement un espace unique de 30 Gio. Un lancement DDP classique réplique généralement le modèle et les états d'entraînement sur chaque GPU ; il augmente le débit, mais ne résout pas à lui seul le dépassement mémoire d'un modèle 1B. Pour répartir réellement les poids, gradients et états d'optimiseur, il faudra vérifier et utiliser un sharding compatible avec fairseq2, par exemple FSDP, ou un offload approprié.

Les T4 utiliseront FP16, activation checkpointing et des microbatches limités par la durée. BF16 ne sera pas sélectionné pour cette configuration.

Avant l'entraînement réel, le notebook exécutera un test de 5 à 20 mises à jour sur les deux GPU et consignera :

- la mémoire maximale de chaque GPU ;
- la durée moyenne d'une mise à jour ;
- la durée estimée de 500 et 5 000 pas ;
- la taille du checkpoint complet ;
- l'espace disque restant après sauvegarde.

Si 500 pas ne peuvent pas tenir dans une session de 12 heures avec une marge suffisante pour l'évaluation et la sauvegarde, l'intervalle sera réduit à 250 pas. Le choix de 500 reste la valeur par défaut tant que le test de débit la confirme.

Le disque devra conserver au maximum :

- le dernier checkpoint complet nécessaire à la reprise ;
- le meilleur export `model-only` connu ;
- les métriques, configurations et prédictions ;
- éventuellement un checkpoint précédent tant que le nouveau n'a pas été vérifié.

Un ancien checkpoint complet ne sera supprimé qu'après rechargement réussi du nouveau et persistance d'une copie hors de l'espace éphémère de la session.

## 13. Plan d'expériences

Le protocole évitera de modifier simultanément les données, les augmentations et les hyperparamètres, car il deviendrait impossible de savoir quelle modification produit l'amélioration.

### Exécution interactive par blocs sur Kaggle

Le mode Kaggle par défaut n'exécutera pas aveuglément les 5 000 pas en une seule cellule. L'entraînement sera organisé en blocs de 500 pas :

```text
charger ou reprendre l'état complet
→ entraîner 500 pas supplémentaires
→ sauvegarder le checkpoint complet
→ valider sur Waxal et Klayt
→ transcrire l'audio témoin
→ afficher et archiver les résultats
→ attendre la décision de lancer le bloc suivant
```

Une fonction ou cellule `run_next_stage()` déterminera automatiquement la prochaine cible — 500, 1 000, 1 500 pas, etc. — et reprendra l'optimiseur, le scheduler, le compteur de pas et le scaler de précision mixte. Le run de 1 000 pas ne devra pas repartir du modèle Meta comme dans l'ancienne expérience.

À la fin de chaque bloc, l'utilisateur pourra :

- poursuivre avec 500 pas supplémentaires ;
- arrêter provisoirement et reprendre dans une autre session ;
- sélectionner le checkpoint déjà sauvegardé ;
- lancer l'export des poids `model-only` puis leur conversion ou publication Hugging Face.

L'arrêt entre deux blocs est préférable à une interruption au milieu d'une mise à jour. Si la cellule est arrêtée accidentellement, seul le dernier checkpoint complet vérifié sera considéré comme récupérable.

### Étape A — Baselines gelées

Évaluer sur les mêmes tests :

- `omniASR_CTC_1B` sans adaptation ;
- `Tree-AI-lab/omniASR-CTC-300M-baoule-1000steps` ;
- éventuellement le checkpoint 1B `_v2` dans une ligne séparée, sans le confondre avec le modèle choisi.

### Étape B — Recette contrôlée sans augmentation

Fine-tuner CTC-1B sur les exemples propres et alignés du mélange, avec les paramètres proches de la recette officielle. Cette étape mesure l'effet du nouveau modèle et des données supplémentaires.

### Étape C — Stratégie de mélange

Comparer un nombre limité de ratios de sources sur la même initialisation et le même budget de pas. Les validations Waxal et Klayt seront examinées séparément.

### Étape D — Robustesse

Ajouter les exemples `noisy_aligned` et les augmentations audio contrôlées. Comparer aux résultats de l'étape C pour vérifier que la robustesse ne s'accompagne pas d'une dégradation excessive sur la parole propre.

### Étape E — Confirmation

Relancer la meilleure configuration avec d'autres seeds si le budget le permet, puis évaluer une seule fois les checkpoints finalistes sur les tests gelés.

### Étape F — Stabilisation du checkpoint

Si plusieurs checkpoints voisins présentent des scores proches, une moyenne des poids des meilleurs checkpoints pourra être comparée au meilleur checkpoint individuel. Cette opération restera une expérience identifiée et ne remplacera pas silencieusement le modèle sélectionné. Elle ne sera conservée que si elle améliore les deux validations sans détériorer la stabilité numérique.

## 14. Sélection des checkpoints

Le checkpoint ne sera pas choisi uniquement sur le WER global concaténé.

Le tableau de validation contiendra au minimum :

- WER et CER Waxal ;
- WER et CER Klayt ;
- moyenne macro des deux WER ;
- moyenne macro des deux CER ;
- pire WER parmi les deux sources ;
- WER/CER par locuteur ;
- taux de sorties vides ;
- substitutions, suppressions et insertions ;
- perte de validation et stabilité de l'entraînement.

Le critère principal proposé est la moyenne macro du WER des deux sources. Le CER et la performance sur la source la plus difficile servent de critères secondaires. Cette méthode évite qu'un corpus plus volumineux détermine seul le checkpoint retenu.

La transcription témoin constitue un outil d'inspection qualitative, pas le critère principal de sélection. Une amélioration visible sur cet audio ne suffit pas à elle seule pour déclarer le checkpoint meilleur ou interrompre définitivement l'expérience.

## 15. Décodage CTC et modèle de langue

Le fine-tuning acoustique et le décodage seront évalués séparément.

La première mesure utilisera toujours le décodage CTC greedy. Elle permet de comparer directement la qualité apprise par le modèle sans apport textuel extérieur.

Une seconde piste pourra utiliser :

- un beam search CTC sans lexique imposé ;
- un modèle de langue baoulé caractère, sous-mot ou mot ;
- une fusion peu profonde entre score CTC et score du modèle de langue ;
- une pénalité d'insertion de mots ajustée sur la validation.

Le corpus textuel servant au modèle de langue devra être nettoyé avec la même politique Unicode que les transcriptions. Les références des tests gelés ne devront jamais être utilisées pour l'entraîner ou régler ses hyperparamètres.

Les résultats finaux distingueront explicitement :

```text
CTC-1B greedy
CTC-1B + beam search
CTC-1B + beam search + modèle de langue
```

Cette séparation permettra d'attribuer correctement les gains au modèle acoustique ou au décodeur. Elle est particulièrement importante lorsque les erreurs portent sur les frontières de mots, les apostrophes et des suites orthographiques proches.

## 16. Évaluation finale

### 16.1 Métriques textuelles

- WER strict après une normalisation minimale documentée ;
- CER strict ;
- WER et CER avec une normalisation orthographique canonique séparée ;
- détail des substitutions, suppressions et insertions ;
- taux de phrases exactes ;
- intervalles de confiance bootstrap à 95 % ;
- différences appariées avec le modèle Tree AI Lab.

Les scores stricts et normalisés seront tous les deux conservés. Il ne faut pas cacher une difficulté orthographique par une normalisation trop agressive.

### 16.2 Découpes d'analyse

Les résultats seront produits :

- par corpus ;
- par locuteur ;
- par durée ;
- par niveau de bruit ou classe de qualité ;
- par présence de silence initial ;
- pour la parole propre et la parole bruitée ;
- pour les segments contenant des mots rares, noms propres ou signes diacritiques.

### 16.3 Performances opérationnelles

- temps de chargement séparé du temps d'inférence ;
- RTF et latence par fichier ;
- médiane et 95e percentile de latence ;
- pic de mémoire GPU ;
- débit selon plusieurs tailles de batch ;
- test du découpage des audios longs avec chevauchement.

### 16.4 Test qualitatif

Une revue humaine examinera un échantillon commun aux modèles :

- meilleures et pires transcriptions ;
- erreurs de frontières de mots ;
- erreurs d'apostrophes et de tons ;
- suppressions de mots courts ;
- erreurs sur noms propres ;
- comportement en présence de bruit ou d'une voix parasite.

## 17. Critère permettant d'affirmer une amélioration

Le CTC-1B fine-tuné sera considéré comme meilleur que le modèle Tree AI Lab seulement si :

1. son WER macro diminue sur les tests gelés ;
2. la différence appariée est accompagnée d'un intervalle de confiance ;
3. le gain n'est pas limité au split de validation Waxal ;
4. aucune source ne subit une régression majeure masquée par l'autre ;
5. les résultats restent cohérents selon les locuteurs ;
6. les contraintes de latence et de mémoire sont mesurées et documentées ;
7. le test final n'a pas servi à choisir les hyperparamètres.

La taille supérieure du modèle ou un meilleur score de développement ne suffisent pas pour conclure.

## 18. Reproductibilité et sauvegardes

Chaque expérience produira :

- les révisions des dépôts et datasets ;
- la configuration YAML complète ;
- le manifeste des exemples retenus et rejetés, avec la raison ;
- les statistiques par corpus et split ;
- la seed ;
- les versions Python, PyTorch, CUDA, fairseq2 et OmniASR ;
- les métriques d'entraînement au format JSONL ou CSV ;
- l'état complet du trainer pour reprendre l'expérience ;
- un export `model-only` séparé pour la publication ;
- un export Hugging Face accompagné du tokenizer et de la configuration, après vérification de parité avec le checkpoint fairseq2 ;
- les prédictions individuelles des validations et tests ;
- les empreintes des références évaluées.

Les sorties seront écrites dans un nouveau dossier par expérience. Une reprise ne devra jamais écraser une expérience antérieure.

La publication Hugging Face sera placée dans une cellule séparée et explicite. Elle ne sera déclenchée qu'après le rechargement du modèle exporté, la comparaison de ses logits ou transcriptions avec le checkpoint source et la vérification de l'identifiant du dépôt cible.

## 19. Arrêts et contrôles automatiques

Le futur script interrompra la préparation ou l'entraînement si :

- un split requis est vide ;
- des locuteurs, audios exacts ou identifiants interdits traversent les splits ;
- le tokenizer produit des inconnus non acceptés ;
- le taux de rejet dépasse un seuil nécessitant une inspection ;
- la perte devient non finie ;
- aucun gradient valide n'est observé ;
- le nombre de paramètres entraînables n'est pas celui attendu ;
- le checkpoint sauvegardé ne peut pas être rechargé ;
- la parité entre le modèle en mémoire et le checkpoint rechargé échoue ;
- l'espace disque devient insuffisant.

Un smoke test de quelques mises à jour et une évaluation de quelques fichiers seront obligatoires avant le lancement des 5 000 pas.

## 20. Livrables prévus pour la phase d'implémentation

Après validation du plan, le dossier devra contenir au minimum :

```text
finetuning_omniasr_ctc_1b_baoule/
├── PLAN_FINETUNING_OMNIASR_CTC_1B_BAOULE.md
├── README.md
├── notebooks/
│   └── finetuning_omniasr_ctc_1b_baoule_kaggle.ipynb
├── configs/
│   ├── ctc_1b_smoke.yaml
│   └── ctc_1b_finetune.yaml
├── src/
│   ├── audit_datasets.py
│   ├── prepare_mixture.py
│   ├── normalize_text.py
│   └── evaluate.py
└── reports/
    └── .gitkeep
```

La première phase de code sera l'audit et la préparation des données. Le lancement d'un entraînement coûteux n'interviendra qu'après inspection de son rapport.

## 21. Résumé du protocole

Le nouveau fine-tuning reposera sur cinq principes :

1. **Alignement avant volume** : une voix parasite non transcrite est corrigée ou rejetée, même si cela réduit le nombre d'heures.
2. **Mélange contrôlé** : Waxal et Klayt restent identifiables et sont échantillonnés sans qu'une source écrase l'autre.
3. **Évaluation indépendante** : les tests restent gelés et les résultats sont séparés par domaine et locuteur.
4. **Entraînement reproductible** : versions, états d'optimiseur, manifests et prédictions sont conservés.
5. **Amélioration démontrée** : la comparaison avec le CTC-300M Tree AI Lab utilise les mêmes audios, des métriques appariées et des intervalles de confiance.

Ce protocole vise à exploiter la qualité acoustique de Waxal et la diversité plus difficile de Common Voice sans transformer les défauts d'alignement de ce dernier en bruit d'apprentissage.

## 22. Sources principales

- [Meta — dépôt officiel Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr)
- [Meta — recette de fine-tuning Wav2Vec2/CTC](https://github.com/facebookresearch/omnilingual-asr/tree/main/workflows/recipes/wav2vec2/asr)
- [Meta — configuration CTC recommandée](https://github.com/facebookresearch/omnilingual-asr/blob/main/workflows/recipes/wav2vec2/asr/configs/ctc-finetune-recommendation.yaml)
- [Meta — préparation des données MixtureParquet](https://github.com/facebookresearch/omnilingual-asr/tree/main/workflows/dataprep)
- [Google — WaxalNLP](https://huggingface.co/datasets/google/WaxalNLP)
- [Klayt — Baoulé Common Voice](https://huggingface.co/datasets/Klayt/baoule-common-voice)
