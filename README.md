# Recommandation de films

Projet fil rouge de notre formation Data Scientist.

Moteur de recommandation content-based : on donne un film, il en sort d'autres qui
lui ressemblent. Pas de filtrage collaboratif, on ne se base que sur le contenu des
films eux-mêmes.

## Données

- IMDB : `title.basics`, `title.ratings`, `title.principals`, `name.basics`
- MovieLens : `movies`, `ratings`, `tags`, `links`

La jointure entre les deux se fait par `links.csv` (`movieId` vers `tconst`).

Après nettoyage on garde les films au-dessus du quantile 0.92 en nombre de votes,
soit 26 463 films.

## Features

Trois blocs, normalisés séparément puis pondérés avant concaténation (`hstack`) :

| bloc | vectorisation | poids |
|---|---|---|
| genres | one-hot | 1.0 |
| casting | CountVectorizer | 0.4 |
| tags | TF-IDF | 0.2 |

La similarité cosinus est calculée à la demande, film par film. Une matrice complète
26463 x 26463 ne tient pas en mémoire.

## Tri par note

La note IMDB brute ne suffit pas : un 9.5 sur 12 votes passe devant un 8.2 sur
400 000. On utilise une moyenne bayésienne, `weight_rating` :

```
WR = (v / (v + m)) * R + (m / (v + m)) * C
```

`R` note du film, `v` son nombre de votes, `m` le seuil de votes retenu, `C` la note
moyenne du corpus.

## Modélisation

Trois notebooks dans `notebooks/03-modeling/`, un par itération.

1. **Genres seuls** + tri `weight_rating` et année. Marche sur le genre, ne détecte
   pas les sagas.
2. **+ casting**. Meilleur dès que deux films partagent des acteurs.
3. **+ tags MovieLens**. Version retenue.


## Installation

```bash
git clone https://github.com/BoostMetrics/Recommandation-de-films.git
cd Recommandation-de-films
python -m venv venv_film
venv_film\Scripts\activate
pip install -r requirements.txt
```

Les données ne sont pas versionnées (plusieurs Go). À télécharger dans `data/raw/` :
[IMDB](https://datasets.imdbws.com/), [MovieLens](https://grouplens.org/datasets/movielens/).

Les notebooks s'exécutent dans l'ordre des dossiers.

## Démo

```bash
streamlit run src/app.py
```

Recherche par titre ou par genre, et trois curseurs pour ajuster les poids des blocs
en direct.


## Limites

Le `hstack` unique pénalise les films riches en tags : plus le vecteur a de
composantes non nulles, plus sa norme monte, et plus le cosinus baisse. Prochaine
étape, trois cosinus séparés et somme pondérée des scores.

Pas de personnalisation par utilisateur. Le seuil de votes écarte le cinéma de niche.

## Structure

```
data/                        données brutes (non versionnées)
docs/                        rapport et schémas
notebooks/01-eda/            exploration IMDB et MovieLens
notebooks/02-preprocessing/  nettoyage, jointures, agrégation casting et tags
notebooks/03-modeling/       les trois itérations
notebooks/04-evaluation/
src/                         code réutilisable et app Streamlit
```


