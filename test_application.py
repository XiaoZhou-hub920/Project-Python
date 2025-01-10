import pytest
from application import Document, CorpusBuilder

# Utilisation de pytest fixture pour charger les résultats de recherche dans CorpusBuilder
@pytest.fixture
def corpus_builder_with_query_results():
    cb = CorpusBuilder()

    # Résultats de recherche
    query_results = [
        {
            "titre": "Houston researchers make nasal vaccine that prevents COVID from spreading",
            "auteur": "rednoise",
            "date": "2024-08-11",
            "texte": "Houston researchers make nasal vaccine that prevents COVID from spreading"
        },
        {
            "titre": "NIH-sponsored trial of nasal COVID-19 vaccine opens",
            "auteur": "BothZookeepergame612",
            "date": "2024-12-26",
            "texte": "NIH-sponsored trial of nasal COVID-19 vaccine opens"
        },
        {
            "titre": "Does the updated COVID-19 vaccine protect against the XEC variant?",
            "auteur": "CTVNEWS",
            "date": "2024-10-23",
            "texte": "Does the updated COVID-19 vaccine protect against the XEC variant?"
        },
        {
            "titre": "You're More Likely to Get Heart Issues from COVID-19 Than the Vaccine",
            "auteur": "burtzev",
            "date": "2024-08-31",
            "texte": "You're More Likely to Get Heart Issues from COVID-19 Than the Vaccine"
        },
        {
            "titre": "What the end of the CDC's COVID vaccine access program means for uninsured Americans",
            "auteur": "I_who_have_no_need",
            "date": "2024-08-29",
            "texte": "What the end of the CDC's COVID vaccine access program means for uninsured Americans"
        }
    ]

    # Charger les données dans le corpus
    for result in query_results:
        cb.corpus.append(
            Document(
                titre=result["titre"],
                auteur=result["auteur"],
                date=result["date"],
                texte=result["texte"]
            )
        )
    return cb

# Test de chargement du corpus
def test_corpus_loading(corpus_builder_with_query_results):
    assert len(corpus_builder_with_query_results.corpus) == 5  # Vérifier que le corpus contient 5 documents

# Test des statistiques
def test_compute_statistics(corpus_builder_with_query_results):
    stats = corpus_builder_with_query_results.compute_statistics()
    assert stats["longueur_corpus"] == 5
    assert stats["moyenne_phrases"] > 0
    assert stats["moyenne_mots"] > 0
    assert stats["total_mots"] > 0

# Test de construction de la matrice TF et TF-IDF
def test_construire_vocabulaire_et_tf(corpus_builder_with_query_results):
    tf_matrix = corpus_builder_with_query_results.construire_vocabulaire_et_tf()
    assert tf_matrix.shape[0] == len(corpus_builder_with_query_results.corpus)
    assert tf_matrix.shape[1] > 0

def test_construire_tf_idf(corpus_builder_with_query_results):
    corpus_builder_with_query_results.construire_vocabulaire_et_tf()
    tf_idf_matrix = corpus_builder_with_query_results.construire_tf_idf()
    assert tf_idf_matrix.shape == corpus_builder_with_query_results.tf_matrix.shape

# Test de la fonction de recherche
def test_recherche(corpus_builder_with_query_results):
    # Construire le vocabulaire et la matrice TF-IDF
    corpus_builder_with_query_results.construire_vocabulaire_et_tf()
    corpus_builder_with_query_results.construire_tf_idf()

    # Vérifier si le mot-clé "vaccine" est dans le vocabulaire
    print("Vocabulaire:", corpus_builder_with_query_results.vocabulaire)
    assert "vaccine" in corpus_builder_with_query_results.vocabulaire

    # Tester la recherche avec un mot-clé existant
    resultats = corpus_builder_with_query_results.recherche("vaccine")
    print("Résultats de la recherche :")
    for doc, score in resultats:
        print(f"Titre: {doc.titre}, Texte: {doc.texte}, Score: {score}")

    # Corriger la logique d'assertion, ignorer la casseCorriger la logique d'assertion, ignorer la casse
    assert len(resultats) > 0  # S'assurer qu'il y a des résultats
    assert all(
        "vaccine" in doc.titre.lower() or "vaccine" in doc.texte.lower()
        for doc, _ in resultats
    )

    # Tester la recherche avec un mot-clé inexistant
    no_result = corpus_builder_with_query_results.recherche("mot_inexistant")
    assert len(no_result) == 0


# Test pour un corpus vide
def test_empty_corpus():
    cb = CorpusBuilder()
    cb.build_corpus()
    stats = cb.compute_statistics()
    assert stats["longueur_corpus"] == 0
    assert stats["moyenne_phrases"] == 0
    assert stats["moyenne_mots"] == 0
    assert stats["total_mots"] == 0
