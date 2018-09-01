from compressed_main import tokenize_and_stem
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabaz_score, pairwise_distances, silhouette_score
import os
from extractor import extract
from compressed_main import stem
import hdbscan
import lda
from stopwords import stop_word_list
import pygal

stopwords = stop_word_list()



def load_text(file_name):
    '''
    loads text from specific folder on the filing system.


      Parameters
    ----------
    file_name: str
        the name of the folder containing the texts

    Returns
    ----------
    total_text : list
        list of raw texts from each document.
    file_names : list
        list of file names.

    '''
    total_text = []
    file_names = []

    path = 'uploads/extracted/' + str(file_name)

    for filename in os.listdir(path):
        text, tokens, keywords = extract(os.path.join(path, filename))
        total_text.append(text)
        file_names.append(filename)

    return total_text, file_names


def tf_idf(total_text):

    '''
    Applies the term-frequency-inverse-document-frequency (TFIDF) algorithm to the input text.


    Parameters
    ----------
    total_text: list[str]
        a list of the input texts.

    Returns
    ----------
    tfidf_matrix : Numpy sparse matrix 
        The vectorised form of the texts.
    '''

    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                min_df=1, stop_words='english',
                                use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
    tfidf_matrix = tfidf_vectorizer.fit_transform(total_text)

    return tfidf_matrix

def count_vec(total_text): 
    '''
    Applies the ScikitLearn CountVectorizer to input text to generate the document-term matrix. 


    Parameters
    ----------
    total_text: list[str]
        a list of the input texts.

    Returns
    ----------
    tdm : Numpy dense matrix 
        Document-term matrix.
    '''
    cvectorizer = CountVectorizer(min_df=1, stop_words=stopwords,  lowercase = True, ngram_range = (1,3))
    tdm = cvectorizer.fit_transform(total_text)
    return tdm


def k_means(total_text, num_clusters, tfidf_matrix):

    '''
    Applies the k-means algorithm to input text and calculates the Silhoutette and Calinski-Harabaz scores.

    Parameters
    ----------
    total_text: list[str]
        a list of the input texts.
    num_clusters: integer
        number of clusters paramter of k-means.
    tfidf_matrix: numpy sparse matrix
        the tf-idf matrix.

    Returns
    ----------
    silhouette: float
        The Silhouette score.
    calinski: float
        The Calinski-Harabaz score.
    '''


    dist = 1 - cosine_similarity(tfidf_matrix)

    km = KMeans(n_clusters=num_clusters)

    cluster_labels = km.fit_predict(tfidf_matrix)

    silhouette = silhouette_score(tfidf_matrix, cluster_labels)
    calinski = calinski_harabaz_score(tfidf_matrix.toarray(), cluster_labels)  

    return silhouette, calinski


def h_dbscan(total_text, min_cluster_size, tfidf_matrix):

    '''
    Applies the HDBSCAN algorithm to input text and calculates the Silhoutette and Calinski-Harabaz scores.

    Parameters
    ----------
    total_text: list[str]
        a list of the input texts.
    min_cluster_size: integer
       minimum number of points that can represent a cluster.
    tfidf_matrix: numpy sparse matrix
        the tf-idf matrix.

    Returns
    ----------
    silhouette: float
        The Silhouette score.
    calinski: float
        The Calinski-Harabaz score.
    '''


    clusterer = hdbscan.HDBSCAN(min_cluster_size)
    labels = clusterer.fit_predict(tfidf_matrix)
    silhouette = silhouette_score(tfidf_matrix, labels)
    calinski = calinski_harabaz_score(tfidf_matrix.toarray(), labels) 

    return silhouette, calinski


def mylda(total_text, n_topics, dtm):

    '''
    Applies the LDA algorithm to input text and calculates the Silhoutette and Calinski-Harabaz scores.

    Parameters
    ----------
    total_text: list[str]
        a list of the input texts.
    n_topics: integer
        number of topics(clusters) parameter of LDA.
    dtm: numpy dense matrix
        the document-term matrix.

    Returns
    ----------
    silhouette: float
        The Silhouette score.
    calinski: float
        The Calinski-Harabaz score.
    '''

    lda_model = lda.LDA(n_topics, 500)
    X_topics = lda_model.fit_transform(dtm)

    lda_keys = []
    for i in range(X_topics.shape[0]):
        lda_keys += X_topics[i].argmax(),

    silhouette = silhouette_score(dtm, lda_keys)
    calinski = calinski_harabaz_score(dtm.toarray(), lda_keys) 

    return silhouette, calinski


k_means_silhouette =  []
k_means_calinski =  []

hdbscan_silhouette = []
hdbscan_calinski  =  []


lda_silhouette = []
lda_calinski = []



# the 300,500,700,1000 are the names of folders containing 
# 300,500,700,1000 text documents respectively   
for x in [300,500,700,1000]:
    total_text, file_names = load_text(str(x))
    tfidf_matrix = tf_idf(total_text)
    dtm = count_vec(total_text)
    rule_of_thumb = int(round(((len(file_names))/2)**0.5))

    silhouette_kmeans, calinski_kmeans = k_means(total_text,rule_of_thumb,tfidf_matrix)
    silhouette_hdbscan, calinski_hdbascan = h_dbscan(total_text,4,tfidf_matrix)
    silhouette_lda, calinski_lda = mylda(total_text,rule_of_thumb,dtm)

    k_means_silhouette.append(silhouette_kmeans)
    k_means_calinski.append(calinski_kmeans)

    hdbscan_silhouette.append(silhouette_hdbscan)
    hdbscan_calinski.append(calinski_hdbascan)

    lda_silhouette.append(silhouette_lda)
    lda_calinski.append(calinski_lda)

    print(str(x) + ' silhouette_kmeans  = ' + str(silhouette_kmeans) + '  calinski_kmeans = ' + str(calinski_kmeans))
    print(str(x) + ' silhouette_hdbscan  = ' + str(silhouette_hdbscan) + '  calinski_hdbascan = ' + str(calinski_hdbascan))
    print(str(x) + ' silhouette_lda  = ' + str(silhouette_lda) + '  calinski_lda = ' + str(calinski_lda))

    f = open('results.txt','a')
    f.write(str(x) + '  silhouette_kmeans  = ' + str(silhouette_kmeans) + '  calinski_kmeans = ' + str(calinski_kmeans) + '\n')
    f.write(str(x) + '  silhouette_hdbscan  = ' + str(silhouette_hdbscan) + '  calinski_hdbascan = ' + str(calinski_hdbascan) + '\n')
    f.write(str(x) + '  silhouette_lda  = ' + str(silhouette_lda) + '  calinski_lda = ' + str(calinski_lda) + '\n')
    f.close()

line_chart = pygal.Bar()
line_chart.title = 'Silhouette scores for k-means, HDBSCAN and LDA'
line_chart.x_labels = ['300','500','700','1000']
line_chart.add('k-means', k_means_silhouette)
line_chart.add('HDBSCAN', hdbscan_silhouette )
line_chart.add('LDA', lda_silhouette  )
line_chart.render_to_file('silhouette_chart.svg') 

line_chart2 = pygal.Bar()
line_chart2.title = 'Calinski-Harabaz scores for k-means, HDBSCAN and LDA'
line_chart2.x_labels = ['300','500','700','1000']
line_chart2.add('k-means', k_means_calinski)
line_chart2.add('HDBSCAN', hdbscan_calinski )
line_chart2.add('LDA', lda_calinski  )
line_chart2.render_to_file('calinski_chart.svg') 


line_chart.render()
line_chart2.render()






    