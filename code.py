import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text as txt
from sklearn.cluster import KMeans

content = []

# Extracting .txt file contents into cleaned python list. 

with open('muslim_text.txt') as text:
    content_dirty = text.read().lower().split('\n\n') # split by paragraph - each paragraph is a document
    for line in content_dirty:
        lines = line.strip().split('\n')
        
        if (len(lines) > 2):
            
            if (lines[0].startswith('volume')): del lines[0] # Remove volume name and number
            #if (lines[0].startswith('narrated')): del lines[0] # Remove narrator of tradition (will keep for now)
            if (lines[-1].startswith('volume')): del lines[-1] # Remove volume name and number
            
            # we want to store the doucments in a manner that kmeans and ML models can easily understand.
            # So we will represent the text as a tfidf. 
            
            lines = ' '.join(lines)
            content.append(lines)

# We now want to create represnetations for the text documents that make sense to be used in Kmeans and Topic mining
# We will use Tf-IDf representations using sci-kit learn
# we use words as terms and throw away english stopwords
# TODO: remove dates, numbers, and maybe stem?


#add some domain-specific stop words
stop_words = txt.ENGLISH_STOP_WORDS.union(["came", "asked","used", "al", "allah", "prophet", "apostle", "bin", "narrated", "abu", "said", "ibn", "man"])

tfidf_vectorizer = TfidfVectorizer(analyzer='word', stop_words=stop_words, lowercase=True, max_df = .025)

# fit the vectorizer to the vocabulary and idf weights from the original text

tfidf_vectorizer.fit(content)
tfidf_weights = tfidf_vectorizer.transform(content)  # `tfidf_weights now holds the `
vocabulary = tfidf_vectorizer.get_feature_names()

# Now we will run K means on the newly processed data
kmeans = KMeans(n_clusters = 100, random_state = 0)
clustered = kmeans.fit_predict(tfidf_weights)


# `clustered` holds the closest center for each sample - we now have the main, primary level categories.
# now create a dictionary of clusters w/actual content

clusters_grouped = {}
for index, value in enumerate(clustered):
    if value in clusters_grouped:
        clusters_grouped[value].append((index, content[index]))
    else:
        clusters_grouped[value] = [(index, content[index])]
        

# `clusters grouped` now holds each cluster's collection of documents.
# we now need to topic mine each cluster and display the top K topics for each cluster

categories = {}
for cluster_num, cluster_docs in clusters_grouped.items():
    
    # Heres the thing. Instead of doing a word count normalization as the weight for each term, we can use the
    # previously calculated tfidf as the term weights instead. Much better approach. Now we just need to
    # normalize the the wceights for each term.
    
    # We topic mine each cluster as a single doc
    tfidf_sum = tfidf_weights[cluster_docs[0][0]]
    for doc in cluster_docs[1:]:
        tfidf_sum = np.add(tfidf_sum, tfidf_weights[doc[0]])
    
    tfidf_sum = tfidf_sum.todense().tolist()[0]
    
    topic_weights = []
    topic_sum = sum(tfidf_sum)
    
    for i, tw in enumerate(tfidf_sum):
        topic_weights.append((i,  float(tw) / float(topic_sum)))
    
    topic_weights = sorted(topic_weights, key = lambda tup: tup[1], reverse = True)
    top = vocabulary[topic_weights[0][0]]
    categories[top] = []
    for topic in topic_weights[1:11]:
        categories[top].append(vocabulary[topic[0]])
        
        
# Final Print

for category, sub_categories in categories.items():
    print(category)
    for sub in sub_categories:
        print('\t' + sub)
    print()
        
        
        
        
        
        
        
        
        