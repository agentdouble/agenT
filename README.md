# agenT

`agenT` est un projet exploratoire deep tech autour d'un "microscope logiciel" pour datasets d'IA.

L'objectif est de rendre la qualite des donnees visible, mesurable et actionnable avant l'entrainement ou le fine-tuning d'un modele.

## Idee

Les performances d'un modele dependent souvent moins du volume brut de donnees que de leur qualite reelle :

- doublons et quasi-doublons ;
- labels incoherents ou probablement faux ;
- classes sous-representees ;
- biais caches dans les donnees ;
- outliers utiles ou dangereux ;
- zones du dataset ou le modele risque d'echouer.

`agenT` vise a analyser ces problemes automatiquement, puis a proposer une liste d'actions concrete pour ameliorer le dataset.

## V0 disponible

La v0 fournit une CLI Python sans dependance externe pour analyser un dataset CSV.

Elle detecte :

- les valeurs manquantes ;
- les lignes exactement dupliquees ;
- les outliers numeriques avec la methode IQR ;
- la distribution des labels ;
- les conflits de labels sur un meme texte ;
- les quasi-doublons textuels simples ;
- une liste d'actions recommandees.

## Utilisation

Analyser le dataset d'exemple :

```bash
python3 -m agent.cli examples/sample_dataset.csv
```

Ecrire le rapport dans un fichier JSON :

```bash
python3 -m agent.cli examples/sample_dataset.csv --output report.json
```

Specifier explicitement les colonnes :

```bash
python3 -m agent.cli examples/sample_dataset.csv --text-column text --label-column label
```

## Tests

```bash
python3 -m unittest
```

## Pourquoi c'est important

Beaucoup d'equipes ameliorent leurs modeles en ajoutant du compute, des parametres ou plus de donnees. Pourtant, un dataset mal annote, redondant ou desequilibre peut limiter fortement les performances.

Un outil qui montre ou les donnees sont faibles peut reduire les couts d'entrainement, accelerer les iterations et rendre les systemes d'IA plus fiables.

## Pistes techniques

- Embeddings multimodaux.
- Detection de similarite et de quasi-duplication.
- Clustering et visualisation de voisinage.
- Estimation d'incertitude.
- Detection d'anomalies.
- Priorisation des corrections par impact attendu.

## Statut

Le projet dispose d'une v0 locale pour CSV. Les prochaines etapes naturelles sont l'ajout d'embeddings, d'une interface d'inspection et d'un score d'impact par correction proposee.
