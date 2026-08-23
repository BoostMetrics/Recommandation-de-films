import streamlit as st
import pickle
import pandas as pd
from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Recommandation de films", layout="wide")

@st.cache_resource
def load_data():
    with open('data_reco.pkl', 'rb') as f:
        return pickle.load(f)

data = load_data()
cast_final_Q92 = data['cast_final_Q92']
genre_cols = data['genre_cols']
genre_matrice = data['genre_matrice']
cast_matrice = data['cast_matrice']
tags_matrice = data['tags_matrice']

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