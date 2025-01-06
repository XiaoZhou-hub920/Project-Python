import praw
import urllib.request
import xmltodict
import numpy as np
import pickle
import datetime
import re
from collections import Counter
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

class Document:
    def __init__(self, titre="", auteur="", date="", url="", texte=""):
        self.titre = titre
        self.auteur = auteur
        self.date = date
        self.url = url
        self.texte = texte

    def __repr__(self):
        return f"Titre : {self.titre}\tAuteur : {self.auteur}\tDate : {self.date}\tURL : {self.url}\tTexte : {self.texte}\t"

    def __str__(self):
        return f"{self.titre}, par {self.auteur}"

class CorpusBuilder:
    def __init__(self):
        self.textes_reddit = []
        self.textes_arxiv = []
        self.corpus = []
        self.vocabulaire = []
        self.tf_matrix = None
        self.tf_idf_matrix = None

    def fetch_reddit_data(self, client_id, client_secret, user_agent, subreddit="Coronavirus", limit=100):
        """
        Collecte les données du subreddit spécifié en utilisant l'API Reddit.
        """
        reddit = praw.Reddit(client_id='wiQgc72eQSxNz0bB2Km-xw', client_secret='bzlnKqXlJltn2zixvCXxPc0_f3aHTg', user_agent='BEYE')
        subr = reddit.subreddit(subreddit)

        for post in subr.hot(limit=limit):
            texte = post.title.replace("\n", " ")
            self.textes_reddit.append(Document(titre=post.title, texte=texte, auteur=str(post.author), date=datetime.datetime.fromtimestamp(post.created_utc).strftime("%Y-%m-%d")))

    def fetch_arxiv_data(self, query="covid", max_results=100):
        """
        Collecte les données d'Arxiv en fonction de la requête spécifiée.
        """
        url = f'http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}'
        url_read = urllib.request.urlopen(url).read()
        data = url_read.decode()
        dico = xmltodict.parse(data)

        docs = dico['feed']['entry'] if isinstance(dico['feed']['entry'], list) else [dico['feed']['entry']]
        for d in docs:
            texte = f"{d['title']}. {d['summary']}".replace("\n", " ")
            auteur = d.get('author', {}).get('name', 'Unknown') if isinstance(d.get('author'), dict) else ', '.join(a['name'] for a in d['author'])
            self.textes_arxiv.append(Document(titre=d['title'], texte=texte, auteur=auteur, date=d.get('published', 'Unknown')))

    def build_corpus(self):
        """
        Combine les données collectées de Reddit et d'Arxiv en un corpus unique.
        """
        self.corpus = self.textes_reddit + self.textes_arxiv
        return self.corpus

    def compute_statistics(self):
        """
        Calcule des statistiques sur le corpus, telles que la moyenne et le total de mots et phrases.
        """
        nb_phrases = [len(doc.texte.split(".")) for doc in self.corpus]
        nb_mots = [len(doc.texte.split(" ")) for doc in self.corpus]

        stats = {
            "longueur_corpus": len(self.corpus),
            "moyenne_phrases": np.mean(nb_phrases) if nb_phrases else 0,
            "moyenne_mots": np.mean(nb_mots) if nb_mots else 0,
            "total_mots": np.sum(nb_mots),
        }
        return stats

    def construire_vocabulaire_et_tf(self):
        """
        Construit le vocabulaire et la matrice TF (Term Frequency) pour le corpus.
        """
        vocabulaire = {}
        rows, cols, data = [], [], []

        for doc_id, doc in enumerate(self.corpus):
            mots = re.findall(r'\b\w+\b', doc.texte.lower())
            compte_mots = Counter(mots)

            for mot, freq in compte_mots.items():
                if mot not in vocabulaire:
                    vocabulaire[mot] = len(vocabulaire)

                rows.append(doc_id)
                cols.append(vocabulaire[mot])
                data.append(freq)

        self.vocabulaire = vocabulaire
        self.tf_matrix = csr_matrix((data, (rows, cols)), shape=(len(self.corpus), len(vocabulaire)))
        return self.tf_matrix

    def construire_tf_idf(self):
        """
        Construit la matrice TF-IDF pour le corpus.
        """
        n_docs = self.tf_matrix.shape[0]
        idf = np.log((1 + n_docs) / (1 + (self.tf_matrix > 0).sum(axis=0))) + 1
        self.tf_idf_matrix = self.tf_matrix.multiply(idf.A1)
        return self.tf_idf_matrix

    def recherche(self, requete, top_n=5):
        """
        Recherche les documents correspondant aux mots-clés de la requête en utilisant la similarité cosinus.
        Retourne les `top_n` résultats avec une mise en forme améliorée.
        """
        mots_recherche = re.findall(r'\b\w+\b', requete.lower())
        vecteur_requete = np.zeros(len(self.vocabulaire))

        for mot in mots_recherche:
            if mot in self.vocabulaire:
                vecteur_requete[self.vocabulaire[mot]] += 1

        scores = []
        for doc_id in range(self.tf_idf_matrix.shape[0]):
            doc_vecteur = self.tf_idf_matrix.getrow(doc_id).toarray().flatten()
            score = 1 - cosine(vecteur_requete, doc_vecteur) if np.any(doc_vecteur) else 0
            scores.append((doc_id, score))

        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        resultats = [(self.corpus[doc_id], score) for doc_id, score in scores if score > 0]

        return resultats[:top_n]

    def export_results(self, results):
        """
        Exporte les résultats de la recherche en CSV.
        """
        data = [{
            "Titre": doc.titre,
            "Auteur": doc.auteur,
            "Date": doc.date,
            "Score": score,
            "Extrait": doc.texte[:200]
        } for doc, score in results]
        return pd.DataFrame(data)

# Application Streamlit
st.title("Moteur de Recherche - Corpus COVID")

st.sidebar.header("Paramètres")
subreddit = st.sidebar.text_input("Subreddit", "Coronavirus")
requete = st.text_input("Entrez votre requête", "")

tri_option = st.sidebar.selectbox("Trier les résultats par", ["Pertinence (Score)", "Date", "Auteur"])

if st.sidebar.button("Collecter les données"):
    builder = CorpusBuilder()
    builder.fetch_reddit_data(client_id="***", client_secret="***", user_agent="***", subreddit=subreddit)
    builder.fetch_arxiv_data()
    builder.build_corpus()
    builder.construire_vocabulaire_et_tf()
    builder.construire_tf_idf()

    st.session_state["builder"] = builder
    st.success("Données collectées avec succès !")

if "builder" in st.session_state:
    builder = st.session_state["builder"]

    # Affichage des statistiques
    stats = builder.compute_statistics()
    st.sidebar.subheader("Statistiques du Corpus")
    st.sidebar.write(f"**Nombre de documents**: {stats['longueur_corpus']}")
    st.sidebar.write(f"**Total de mots**: {stats['total_mots']}")
    st.sidebar.write(f"**Moyenne de phrases**: {stats['moyenne_phrases']:.2f}")

    if requete:
        st.header(f"Résultats pour : {requete}")
        resultats = builder.recherche(requete)

        if resultats:
            # Tri des résultats
            if tri_option == "Date":
                resultats.sort(key=lambda x: x[0].date, reverse=True)
            elif tri_option == "Auteur":
                resultats.sort(key=lambda x: x[0].auteur)

            # Graphique de scores
            scores = [score for _, score in resultats]
            titres = [doc.titre[:30] for doc, _ in resultats]
            fig, ax = plt.subplots()
            ax.barh(titres[::-1], scores[::-1])
            ax.set_xlabel("Scores de Similarité")
            ax.set_title("Scores des Résultats")
            st.pyplot(fig)

            # Affichage des résultats
            for i, (doc, score) in enumerate(resultats):
                st.subheader(f"{i+1}. {doc.titre}")
                st.write(f"**Auteur** : {doc.auteur}")
                st.write(f"**Date** : {doc.date}")
                st.write(f"**Score** : {score:.4f}")
                st.write(f"**Extrait** : {doc.texte[:200]}...")

            # Export des résultats
            df_results = builder.export_results(resultats)
            csv = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Télécharger les résultats au format CSV",
                data=csv,
                file_name="resultats_recherche.csv",
                mime="text/csv"
            )
        else:
            st.warning("Aucun résultat trouvé.")
