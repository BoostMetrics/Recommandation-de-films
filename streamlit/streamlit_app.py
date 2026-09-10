from pathlib import Path
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from tabulate import tabulate
import sklearn.metrics.pairwise as dist
import pickle

DATA = Path(".")

st.set_page_config(page_title="Reco films", page_icon="🎬", layout="wide")


# --------------------------------------------------------------------------- #
# Chargement (mis en cache)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_movies() -> pd.DataFrame:
    df = pd.read_csv(DATA / "movies_cleaned.csv")
    df["year"] = df["title"].str.extract(r"\((\d{4})\)\s*$").astype(float)
    return df

@st.cache_data(show_spinner=False)
def load_ratings() -> pd.DataFrame:
    df = pd.read_csv(DATA / "ratings_cleaned.csv")
    return df

@st.cache_data(show_spinner=False)
def load_tags() -> pd.DataFrame:
    df = pd.read_csv(DATA / "tags_cleaned.csv")
    return df

@st.cache_data(show_spinner=False)
def load_genome_scores() -> pd.DataFrame:
    df = pd.read_csv(DATA / "genome_scores_cleaned.csv")
    return df

@st.cache_data(show_spinner=False)
def load_genome_tags() -> pd.DataFrame:
    df = pd.read_csv(DATA / "genome_tags_movies.csv")
    return df

@st.cache_data(show_spinner=False)
def load_common_movieId() -> pd.DataFrame:
    df = pd.read_csv(DATA / "genome_tags_movies.csv")
    common_movieIds = set(movies['movieId']) & set(ratings['movieId']) & set(tags['movieId']) & set(genome_scores['movieId']) & set(genome_tags['movieId'])
    df = pd.DataFrame({'movieId': list(common_movieIds)}).merge(
        movies[['movieId', 'title']],
        on='movieId',
        how='left'
    )
    return df

movies = load_movies()
ratings = load_ratings()
tags = load_tags()
genome_scores = load_genome_scores()
genome_tags = load_genome_tags()
film_recherche = load_common_movieId()


GENRES = sorted(c[6:] for c in movies.columns if c.startswith("genre_"))

st.sidebar.title("Sommaire")
PAGES = ["Données", "MovieLens - Recommandation film", "MovieLens - Recommandation utilisateur"]
page = st.sidebar.radio("Aller vers", PAGES)
st.sidebar.markdown("---")
st.sidebar.caption(
    "MovieLens 20M · 20 M notes · 138 k utilisateurs · 27 k films\n\n"
    "IMDB"
)

# --------------------------------------------------------------------------- #
# Fonction pour la modélisation
# --------------------------------------------------------------------------- #
indices = pd.Series(range(0, len(tags)), index=tags['movieId'])

@st.cache_data(show_spinner=False)
def modele1(tags):
    tfidf = TfidfVectorizer(stop_words='english')
    matrice_tfidf = tfidf.fit_transform(tags['tag'])
    sim_cosinus = cosine_similarity(matrice_tfidf, matrice_tfidf)
    sim_euclidienne = 1 / (1 + euclidean_distances(matrice_tfidf))
    return sim_cosinus, sim_euclidienne

sim_cosinus, sim_euclidienne = modele1(tags)

def recommander_films(movie_id, mat_sim=sim_cosinus, num_recommendations=10):

    # Vérifier que le movieId existe
    if movie_id not in indices.index:
        print(f"Aucun film trouvé pour movieId={movie_id}")
        return None
    
    # Récupérer la position dans la matrice
    idx = indices[movie_id]
    titre_film = tags.iloc[idx]['title']

    print(f"Films recommandés pour : {titre_film} \n")

    # Obtenir et trier les scores de similarité
    scores_similarite = sorted(
        enumerate(mat_sim[idx]), key=lambda x: x[1], reverse=True
    )

    # Top N films similaires (hors le film lui-même)
    top_similair = scores_similarite[1:num_recommendations + 1]
    res = [(tags.iloc[i]['title'], score) for i, score in top_similair]

    return tabulate(res, headers=["Titre", "Score de similarité"], tablefmt="pretty")



indices_genome = pd.Series(range(len(genome_tags)), index=genome_tags['movieId'])

@st.cache_data(show_spinner=False)
def modele2(genome_tags):
    tfidf_genome = TfidfVectorizer(stop_words='english')
    matrice_tfidf_genome = tfidf_genome.fit_transform(genome_tags['tag'])
    sim_cosinus_genome = cosine_similarity(matrice_tfidf_genome, matrice_tfidf_genome)
    sim_euclidienne_genome = 1 / (1 + euclidean_distances(matrice_tfidf_genome))
    return sim_cosinus_genome, sim_euclidienne_genome

sim_cosinus_genome, sim_euclidienne_genome = modele1(genome_tags)

def recommander_films_genome(movie_id, mat_sim=sim_cosinus_genome, num_recommendations=10):
    # Vérifier que le movieId existe
    if movie_id not in indices_genome.index:
        print(f"Aucun film trouvé pour movieId={movie_id}")
        return None

    # Récupérer la position dans la matrice de similarité
    idx = indices_genome[movie_id]
    titre_film = genome_tags.iloc[idx]['title']

    print(f"Films recommandés pour : {titre_film} (movieId={movie_id})\n")

    scores_similarite = sorted(
        enumerate(mat_sim[idx]), key=lambda x: x[1], reverse=True
    )

    # Top N films similaires (hors le film lui-même)
    top_similair = scores_similarite[1:num_recommendations + 1]
    res = [(genome_tags.iloc[i]['title'], score) for i, score in top_similair]
    return tabulate(res, headers=["Titre", "Score de similarité"], tablefmt="pretty")

# approche filtrage collaboratif : item-based filtering

@st.cache_data
def load_demo():
    with open('demo_recommandations.pkl', 'rb') as f:
        return pickle.load(f)

resultats_demo = load_demo()

# --------------------------------------------------------------------------- #
# 1. DONNEES
# --------------------------------------------------------------------------- #
if page == "Données":
    st.title("🎬 Recommandation de films")
    st.header("Données utilisées")

    st.subheader("`MOVIELENS`")
    c1, c2, c3 = st.columns(3)
    c1.metric("Films (catalogue)", f"{len(movies):,}".replace(",", " "))
    c2.metric("Films notés", f"{ratings['movieId'].nunique():,}".replace(",", " "))
    c3.metric("Films tagués", f"{len(tags):,}".replace(",", " "))
    

    c4, c5, c6 = st.columns(3)
    c4.metric("Notes (utilisateurs)", f"{len(ratings):,}".replace(",", " "))
    c5.metric("Utilisateurs", f"{ratings['userId'].nunique():,}".replace(",", " "))
    c6.metric("", "")

    st.subheader("`IMDB`")

    st.subheader("Fichiers MovieLens du projet")
    st.markdown(
        "- `movies_cleaned.csv` — genres binarisés (`movieId`)\n"
        "- `ratings_cleaned.csv` — 20 M notes (`userId`, `movieId`, `rating`)\n"
        "- `tags_cleaned.csv` — tags libres des utilisateurs agrégés par film (`movieId`, `title`, `tag`)\n"
        "- `genome_scores_cleaned.csv` — score (relevance ≥ 0.5) des genome tags  par film (`movieId`, `tagId`, `relevance`)\n"
        "- `genome_tags_movies.csv` — genome tags agrégés par film (`movieId`, `title`, `tag`)\n"
    )


# --------------------------------------------------------------------------- #
# 2. MOVIELENS - RECOMMANDATION FILM
# --------------------------------------------------------------------------- #
elif page == "MovieLens - Recommandation film":
    st.title("Recommandation de film")

    st.subheader("`Recommandation d'un film similaire à un film`")

    saisie = st.text_input("Titre du film", placeholder="Ex : Matrix")

    if saisie:
        resultats = film_recherche[film_recherche["title"].str.contains(saisie, case=False, na=False)]

        if resultats.empty:
            st.warning("Aucun film ne correspond à votre saisie.")
        else:
            propositions = resultats.head(10)
            choix = st.radio(
                f"{len(propositions)} proposition(s) :",
                propositions["title"].tolist(),
            )

            film = propositions[propositions["title"] == choix].iloc[0]
            movieId = film['movieId']
            #st.write(movieId)
            #st.success(f"movieId : {film['movieId']}")

            # Filtrage sur le contenu : utilisattion des tags utilisateurs pour déterminer la similarité entre les films
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Filtrage TAG USER — Similarité cosinus**")
                st.code(recommander_films(movieId, sim_cosinus))
                st.write("**Filtrage TAG USER — Similarité euclidienne**")
                st.code(recommander_films(movieId, sim_euclidienne))

            with col2:
                st.write("**Filtrage TAG GENOME — Similarité cosinus**")
                st.code(recommander_films_genome(movieId, sim_cosinus_genome))
                st.write("**Filtrage TAG GENOME — Similarité euclidienne**")
                st.code(recommander_films_genome(movieId, sim_euclidienne_genome))


# --------------------------------------------------------------------------- #
# 3. MOVIELENS - RECOMMANDATION UTILISATEUR
# --------------------------------------------------------------------------- #
elif page == "MovieLens - Recommandation utilisateur":
    st.title("Recommandation de film")

    st.subheader("`Recommandation d'un film à un utilisateur`")

    user_id = st.selectbox("Choisir un utilisateur", options=list(resultats_demo.keys()))

    films_notes = ratings[ratings['userId'] == user_id].merge(
        movies[['movieId', 'title']], on='movieId', how='left'
    )[['title', 'rating']].sort_values('rating', ascending=False)

    st.subheader("FIlms notés par l'utilisateur")
    st.dataframe(films_notes, use_container_width=True)

    st.subheader("Recommandations")
    df_trie = resultats_demo[user_id].sort_values('note_predite', ascending=False).head(20)
    st.dataframe(df_trie, use_container_width=True, hide_index=True)
    