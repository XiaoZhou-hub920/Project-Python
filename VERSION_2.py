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

        if not resultats:
            return "Aucun résultat trouvé pour votre requête."

        # Mise en forme des résultats
        output = "\nTop résultats pour votre requête :\n"
        for i, (doc, score) in enumerate(resultats[:top_n]):
            extrait = doc.texte[:200] + "..." if len(doc.texte) > 200 else doc.texte
            output += f"\n{i+1}. {doc.titre} (Score: {score:.4f})\n   Auteur: {doc.auteur}\n   Date: {doc.date}\n   Extrait: {extrait}\n"
        return output

if __name__ == "__main__":
    builder = CorpusBuilder()

    # Fetch data (replace placeholders with actual Reddit credentials)
    builder.fetch_reddit_data(client_id="***identifiant***", client_secret="***motdepasse***", user_agent="***nom***")
    builder.fetch_arxiv_data()

    # Build and analyze corpus
    corpus = builder.build_corpus()
    stats = builder.compute_statistics()

    print(f"Longueur du corpus : {stats['longueur_corpus']}")
    print(f"Moyenne du nombre de phrases : {stats['moyenne_phrases']:.2f}")
    print(f"Moyenne du nombre de mots : {stats['moyenne_mots']:.2f}")
    print(f"Nombre total de mots dans le corpus : {stats['total_mots']}")

    # Construire vocabulaire, matrice TF, et matrice TF-IDF
    builder.construire_vocabulaire_et_tf()
    builder.construire_tf_idf()

    # Recherche dans le corpus
    requete = "covid vaccine"
    resultats = builder.recherche(requete)
    print(resultats)

    # Display current date
    print(f"Date actuelle : {datetime.datetime.now()}")
