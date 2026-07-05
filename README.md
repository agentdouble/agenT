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

## Version minimale

Une premiere version peut se concentrer sur un workflow simple :

1. Charger un dataset texte, image ou tabulaire.
2. Generer des embeddings pour chaque exemple.
3. Detecter les clusters, doublons, outliers et incoherences de labels.
4. Afficher une vue d'inspection claire.
5. Proposer des actions : supprimer, relabeler, fusionner, reequilibrer ou collecter plus de donnees.

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

Le projet est au stade de cadrage initial.
