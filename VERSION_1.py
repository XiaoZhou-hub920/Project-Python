import praw
import urllib.request
import xmltodict
import numpy as np
import pickle
import datetime

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

class Author:
    def __init__(self, name):
        self.name = name
        self.ndoc = 0
        self.production = []

    def add(self, production):
        self.ndoc += 1
        self.production.append(production)

    def __str__(self):
        return f"Auteur : {self.name}\t# productions : {self.ndoc}"

class CorpusBuilder:
    def __init__(self):
        self.textes_reddit = []
        self.textes_arxiv = []
        self.corpus = []

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

    def filter_long_texts(self, min_length=100):
        """
        Filtre les documents ayant un texte supérieur à une longueur minimale spécifiée.
        """
        return [doc for doc in self.corpus if len(doc.texte) > min_length]

    def save_corpus(self, corpus, filename="out.pkl"):
        """
        Sauvegarde le corpus filtré dans un fichier pickle.
        """
        with open(filename, "wb") as f:
            pickle.dump(corpus, f)

    def load_corpus(self, filename="out.pkl"):
        """
        Charge un corpus à partir d'un fichier pickle.
        """
        with open(filename, "rb") as f:
            return pickle.load(f)

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

    # Filter and save corpus
    filtered_corpus = builder.filter_long_texts()
    builder.save_corpus(filtered_corpus)

    # Display current date
    print(f"Date actuelle : {datetime.datetime.now()}")
