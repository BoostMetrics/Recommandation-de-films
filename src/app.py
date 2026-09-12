import streamlit as st
import pickle
import pandas as pd
from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

CHEMIN_PKL = Path(__file__).resolve().parent.parent / "data" / "processed" / "data_reco.pkl"

st.set_page_config(page_title="Recommandation de films", layout="wide")

@st.cache_resource
def load_data():
    with open(CHEMIN_PKL, 'rb') as f:
        return pickle.load(f)

data = load_data()
cast_final_Q92 = data['cast_final_Q92']
genre_cols = data['genre_cols']
genre_matrice = data['genre_matrice']
cast_matrice = data['cast_matrice']
tags_matrice = data['tags_matrice']

st.sidebar.title("Sommaire")
PAGES = ["IMDB", "MovieLens - Filtrage continu", "MovieLens - Filtrage collaboratif"]
page = st.sidebar.radio("Aller vers", PAGES)

st.sidebar.header("Pondérations")
w_genre = st.sidebar.slider("Genre", 0.0, 2.0, 1.0, 0.1)
w_cast = st.sidebar.slider("Casting", 0.0, 2.0, 1.0, 0.1)
w_tags = st.sidebar.slider("Tags", 0.0, 2.0, 1.0, 0.1)

@st.cache_resource
def build_matrice(w_genre, w_cast, w_tags):
    return hstack([
        genre_matrice * w_genre,
        cast_matrice * w_cast,
        tags_matrice * w_tags,
    ]).tocsr()

matrice = build_matrice(w_genre, w_cast, w_tags)

def recherche_par_titre(titre, n=10):
    exact = cast_final_Q92[cast_final_Q92['primaryTitle'].str.lower() == titre.lower()]
    if not exact.empty:
        matches = exact
        position = matches.sort_values('numVotes', ascending=False).index[0]
    else:
        matches = cast_final_Q92[cast_final_Q92['primaryTitle'].str.contains(titre, case=False, na=False)]
        if matches.empty:
            return None, None
        position = matches.sort_values('weight_rating', ascending=False).index[0]

    sims = cosine_similarity(matrice[position], matrice).flatten()
    df_temp = cast_final_Q92.copy()
    df_temp['sim'] = sims
    df_temp = df_temp.drop(position)
    df_temp = df_temp.sort_values(['sim'], ascending=[False])

    film_trouve = cast_final_Q92.loc[position, 'primaryTitle']
    return film_trouve, df_temp[['primaryTitle', 'startYear', 'clean_name', 'weight_rating', 'tag', 'sim']].head(n)

def recherche_par_genre(genre, n=10):
    mapping = {g.lower(): g for g in genre_cols}
    genre_col = mapping.get(genre.lower())
    if genre_col is None:
        return None
    films = cast_final_Q92[cast_final_Q92[genre_col] == 1]
    if films.empty:
        return None
    return films.sort_values('weight_rating', ascending=False)[
        ['primaryTitle', 'startYear', 'clean_name', 'weight_rating']
    ].head(n)



#######################################
# AJout partie MovieLens
#######################################



from pathlib import Path
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from tabulate import tabulate
import sklearn.metrics.pairwise as dist

DATA = Path(".")

#st.set_page_config(page_title="Reco films", page_icon="🎬", layout="wide")


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
# 1. MOVIELENS - RECOMMANDATION FILM
# --------------------------------------------------------------------------- #
if page == "MovieLens - Filtrage continu":
    st.title("🎬 Recommandation de films")

    st.subheader("`Filtrage continu - recherche des films similaires`")

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

            # Filtrage sur le contenu : utilisation des tags utilisateurs pour déterminer la similarité entre les films
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
# 2. MOVIELENS - RECOMMANDATION UTILISATEUR
# --------------------------------------------------------------------------- #
elif page == "MovieLens - Filtrage collaboratif":
    st.title("🎬 Recommandation de films")

    st.subheader("`Filtrage collaboratif - recommander sur la base des comportements des autres utilisateurs`")

    user_id = st.selectbox("Choisir un utilisateur", options=list(resultats_demo.keys()))

    films_notes = ratings[ratings['userId'] == user_id].merge(
        movies[['movieId', 'title']], on='movieId', how='left'
    )[['title', 'rating']].sort_values('rating', ascending=False)

    st.subheader("FIlms notés par l'utilisateur")
    st.dataframe(films_notes, use_container_width=True)

    st.subheader("Recommandations")
    df_trie = resultats_demo[user_id].sort_values('note_predite', ascending=False).head(20)
    st.dataframe(df_trie, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# 3. IMDB
# --------------------------------------------------------------------------- #
elif page == "IMDB":
    st.title("🎬 Recommandation de films")

    tab1, tab2 = st.tabs(["Par titre", "Par genre"])

    with tab1:
        titre = st.text_input("Titre du film")
        n1 = st.slider("Nombre de résultats", 5, 20, 10, key="n1")
        if titre:
            film_trouve, resultats = recherche_par_titre(titre, n1)
            if resultats is None:
                st.warning(f"Aucun film trouvé pour '{titre}'")
            else:
                st.caption(f"Film de référence : **{film_trouve}**")
                st.dataframe(resultats, use_container_width=True)

    with tab2:
        genre = st.selectbox("Genre", sorted(genre_cols))
        n2 = st.slider("Nombre de résultats", 5, 20, 10, key="n2")
        resultats_genre = recherche_par_genre(genre, n2)
        if resultats_genre is None:
            st.warning(f"Aucun film trouvé pour '{genre}'")
        else:
            st.dataframe(resultats_genre, use_container_width=True)