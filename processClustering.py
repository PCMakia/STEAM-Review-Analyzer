from functools import reduce
import numpy as np
import json as js
from gensim.parsing.preprocessing import remove_stopwords, STOPWORDS
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import re
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor

# Contraction from stackoverflow
contractions = { 
"ain't": "am not / are not / is not / has not / have not",
"aren't": "are not / am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is / how does",
"i'm" : "i am",
"i'd": "i had / i would",
"i'd've": "i would have",
"i'll": "i shall / i will",
"i'll've": "i shall have / i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have"
}
#----------------------------Word expansion-----------------
# Function to check if a sentence has any contracted word
def has_contraction(sentence):
    # Use a regular expression to check for any contraction from the dictionary
    pattern = r'\b(' + '|'.join(re.escape(key) for key in contractions.keys()) + r')\b'
    return bool(re.search(pattern, sentence.lower()))  # Convert to lowercase for case-insensitive matching

# Function to expand contractions in a sentence
# ----------------------------Too long --------------------------
# def expand_contractions(sentence):
#     words = sentence.split()
#     expanded_sentences = ['']  # Start with an empty sentence to modify
    
#     # Iterate over the words in the sentence
#     for word in words:
#         word = word.lower()  # Convert to lowercase to handle case-insensitive matching
#         if word in contractions:
#             expanded_forms = contractions[word].split(" / ")
#             # For each expansion, create new sentences
#             new_sentences = []
#             for expansion in expanded_forms:
#                 for sentence in expanded_sentences:
#                     new_sentences.append(f"{sentence} {expansion}")
#             expanded_sentences = new_sentences
#         else:
#             # If no contraction is found, just add the word to all sentences
#             expanded_sentences = [f"{sentence} {word}" for sentence in expanded_sentences]
    
#     # Return the expanded sentences
#     return [sentence.strip() for sentence in expanded_sentences]
#-------------------------------------------------------------------------------------------------
def expand_contractions(sentence):
    words = sentence.split()
    expanded_sentence = []  # Start with an empty list for the expanded words

    # Iterate over the words in the sentence
    for word in words:
        lower_word = word.lower()  # Convert to lowercase to handle case-insensitive matching
        if lower_word in contractions:
            # Use the first form of the contraction expansion
            expanded_form = contractions[lower_word].split(" / ")[0]
            expanded_sentence.append(expanded_form)
        else:
            # If no contraction is found, just add the word as is
            expanded_sentence.append(word)
    
    # Join the words to form the final expanded sentence
    return " ".join(expanded_sentence)


# Load pre-trained GloVe model (using Gensim to load GloVe)
def load_glove_model(glove_file_path):
    model = KeyedVectors.load_word2vec_format(glove_file_path, binary=False, no_header=True)
    return model

# Function to clean the sentences (remove non-alphabetic characters)
def clean_sentence(sentence):
    return re.sub(r"[^a-zA-Z\s]", "", sentence).lower().split()

# Function to get the sentence vector by averaging word vectors
def get_sentence_vector(sentence, model):
    words = clean_sentence(sentence)
    word_vectors = []
    
    for word in words:
        if word in model:
            word_vectors.append(model[word])
    
    if len(word_vectors) == 0:  # If no words from the sentence are in the model
        return np.zeros(model.vector_size)
    
    return np.mean(word_vectors, axis=0)

# Function to compare sentences using cosine similarity
def compare_sentences(sentences, model):

    sentence_vectors = [get_sentence_vector(sentence, model) for sentence in sentences]

    similarities = []

    for i in range(len(sentence_vectors)):
        for j in range(i + 1, len(sentence_vectors)):
            # Calculate cosine similarity between sentence vectors
            similarity = cosine_similarity([sentence_vectors[i]], [sentence_vectors[j]])
            similarities.append((i, j, similarity[0][0]))
            
    return similarities


def ParToSen(paragraphs):
    sentences = []
    for paragraph in paragraphs:
        # Split the paragraph into sentences using a regular expression that handles punctuation
        paragraph_sentences = re.split(r'(?<=\.)\s+', paragraph.strip())
        sentences.extend(paragraph_sentences)
    return sentences
# from Similarity import SimGloVe

# Run this block for first time running-----------
# import nltk
# nltk.download('punkt_tab')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('vader_lexicon')
# -------------------V2 REVISION-----------------------
def filter_short_responses(data_block, min_length=5):
    # Ensure the 'response' key exists in the data block
    if "response" not in data_block:
        return data_block  # Return unchanged if no 'response' key is found
    
    # Filter out sentences shorter than min_length
    filtered_responses = [sentence for sentence in data_block["response"] if len(sentence) >= min_length]
    
    # Update the 'response' key with the filtered sentences
    data_block["response"] = filtered_responses

    return data_block

# new processing process, input dictionary of game, output combinatyion dictionary
def process_sentences(review_sentences):

    pos_combinations = {
        "noun-adjective": [],
        "noun-verb": [],
        "noun-adverb": [],
        "verb-adjective": [],
        "verb-adverb": [],
        "adjective-adverb": []
    }

    # Helper function to classify word pairs into categories
    def classify_pair(tag1, tag2):
        sorted_tags = tuple(sorted([tag1, tag2]))  # Sort tags alphabetically
        if sorted_tags == ('JJ', 'NN'):  # Noun-Adjective or Adjective-Noun
            return "noun-adjective"
        elif sorted_tags == ('NN', 'VB'):  # Noun-Verb or Verb-Noun
            return "noun-verb"
        elif sorted_tags == ('NN', 'RB'):  # Noun-Adverb or Adverb-Noun
            return "noun-adverb"
        elif sorted_tags == ('JJ', 'VB'):  # Verb-Adjective or Adjective-Verb
            return "verb-adjective"
        elif sorted_tags == ('RB', 'VB'):  # Verb-Adverb or Adverb-Verb
            return "verb-adverb"
        elif sorted_tags == ('JJ', 'RB'):  # Adjective-Adverb or Adverb-Adjective
            return "adjective-adverb"
        return None

    # Process each sentence
    for sentence in review_sentences:
        # Get the sentiment score for the sentence
        sentiment = get_sentiment(sentence)

        # Tokenize the sentence into words
        tokens = word_tokenize(sentence)
        
        # Get part-of-speech tags for each token
        pos_tags = pos_tag(tokens)
        
        # Identify adjacent pairs based on specified combinations
        for i in range(len(pos_tags) - 1):
            word1, tag1 = pos_tags[i]
            word2, tag2 = pos_tags[i + 1]

            # Determine the category for the pair
            category = classify_pair(tag1[:2], tag2[:2])  # Use the first two letters of tags
            if category:
                # Sort the words alphabetically for consistency in pairs
                sorted_pair = tuple(sorted([word1, word2]))
                pos_combinations[category].append((sentiment, *sorted_pair))

    # Sort each list of pairs alphabetically by the word pairs
    for key in pos_combinations:
        pos_combinations[key] = sorted(pos_combinations[key], key=lambda x: (x[1], x[2]))

    return pos_combinations

#---------------------Clustering by k Mean-------------
# Clean Tokens (Remove unwanted characters like "n't", "'s")
def clean_tokens(tokens):
    return [token.replace("n't", "").replace("'s", "") for token in tokens if token.replace("n't", "").replace("'s", "") != ""]

# Function to get the GloVe vectors for each word in the sentence
def get_word_vectors(tokens, model):
    word_vectors = []
    cleaned_tokens = clean_tokens(tokens)
    
    # Fetch the word vector for each token in GloVe model
    for word in cleaned_tokens:
        if word in model:
            word_vectors.append(model[word])
    
    return word_vectors, cleaned_tokens

# Function to perform KMeans clustering on the word vectors
def cluster_words_with_kmeans(word_vectors, num_clusters=3):
    if len(word_vectors) == 0:
        return "No valid word vectors found for clustering."
    
    # Apply KMeans clustering to group similar words
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(word_vectors)
    
    # Get the clusters
    clusters = kmeans.predict(word_vectors)
    
    return kmeans, clusters

def cluster_word_pairs(word_pairs, glove_model, num_clusters=3):
    if not word_pairs:
        return {"clusters": [], "words": [], "pairs": [], "kmeans": None}

    # Extract the first words from each pair
    first_words = [pair[1] for pair in word_pairs]
    
    # Get GloVe vectors for the first words
    word_vectors, cleaned_tokens = get_word_vectors(first_words, glove_model)
    
    if not word_vectors:
        print("No valid GloVe vectors found for the words.")
        return {"clusters": [], "words": [], "pairs": [], "kmeans": None}
    
    # Perform clustering on the word vectors of the first words
    kmeans, cluster = cluster_words_with_kmeans(word_vectors, num_clusters)
    
    # Organize clustering results
    cluster_results = {}
    for idx, label in enumerate(cluster):
        if label not in cluster_results:
            cluster_results[label] = []
        # Add the original word pair to the cluster
        original_pair = word_pairs[idx]
        cluster_results[label].append(original_pair)

    # Return the clustering results along with the kmeans object for later analysis
    return {
        "clusters": cluster_results,
        "words": cleaned_tokens,
        "pairs": word_pairs,
        "kmeans": kmeans
    }


# Process and cluster
def process_sentences_and_cluster(review_sentences, glove_model, num_clusters=3):
    # Step 1: Extract POS combinations
    pos_combinations = process_sentences(review_sentences)
    print('sentence processed')
    # Step 2: Cluster each POS combination based on the first word in pairs
    clustered_combinations = {}
    for key, word_pairs in pos_combinations.items():
        clustered_combinations[key] = cluster_word_pairs(word_pairs, glove_model, num_clusters)
        print('1 category done')
    return clustered_combinations

#-----------------------------Picking word pairs----------------
def get_word_vector_from_pair(pair, glove_model):
    # Assuming you want to use the first word of the pair for clustering
    word1, word2 = pair[1], pair[2]
    
    # Get GloVe vectors for the words
    vectors, _ = get_word_vectors([word1, word2], glove_model)
    
    if len(vectors) == 2:
        # If we have both word vectors, average them to represent the pair
        return np.mean(vectors, axis=0)
    else:
        # If either word is not found in GloVe, return a zero vector
        return np.zeros(glove_model.vector_size)

# Calculated by Cosine
# def COS_pick_closest_word_pairs_to_centers(clustered_data, glove_model, num_pairs=3):
    selected_pairs = {}

    # Set to track globally selected word pairs to ensure uniqueness
    global_selected_pairs = set()

    # Iterate through each category in clustered_data
    for category, data in clustered_data.items():
        print(f'Calculating a category by COS: {category}')
        
        # Extract the values for the current category
        cluster_results = data["clusters"]
        kmeans = data["kmeans"]

        # Get the cluster centers from kmeans
        centers = kmeans.cluster_centers_

        # Dictionary to store the closest word pairs for each cluster
        cluster_selected_pairs = {}

        # Iterate through each cluster
        for label, word_pairs in cluster_results.items():
            # Get the center of the current cluster
            center = centers[label]

            # List to store the distance of each pair from the center
            distances = []

            # Calculate the distance from each word pair to the cluster center
            for pair in word_pairs:
                # Calculate the word vector
                word_vector = get_word_vector_from_pair(pair, glove_model)

                # Calculate cosine similarity between word pair vector and cluster center
                similarity = cosine_similarity([word_vector], [center])[0][0]

                # Cosine similarity is higher when vectors are closer, so we use (1 - similarity) to get a distance measure
                distance = 1 - similarity
                distances.append((pair, distance))

            # Sort the word pairs by distance to the center (ascending order of distance)
            distances.sort(key=lambda x: x[1])

            # Pick the top `num_pairs` closest unique word pairs
            unique_pairs = []
            for pair, _ in distances:
                if pair not in global_selected_pairs:
                    unique_pairs.append(pair)
                    global_selected_pairs.add(pair)

                # Stop if we have enough unique pairs
                if len(unique_pairs) >= num_pairs:
                    break

            cluster_selected_pairs[label] = unique_pairs

        # Add the selected pairs for the current category
        selected_pairs[category] = cluster_selected_pairs

    return selected_pairs

# def COS_pick_closest_word_pairs_to_centers(clustered_data, glove_model, num_pairs=3):
#     selected_pairs = {}

#     # Iterate through each category in clustered_data
#     for category, data in clustered_data.items():
#         print('Calculating a category by COS')
#         # Extract the values for the current category
#         cluster_results = data["clusters"]
#         kmeans = data["kmeans"]
#         word_pairs = data["pairs"]
#         # cleaned_tokens = data["words"]

#         # Get the cluster centers from kmeans
#         centers = kmeans.cluster_centers_

#         # Dictionary to store the closest word pairs for each cluster
#         cluster_selected_pairs = {}

#         # Iterate through each cluster
#         for label, word_pairs in cluster_results.items():
#             # Get the center of the current cluster
#             center = centers[label]
            
#             # List to store the distance of each pair from the center
#             distances = []
            
#             # Calculate the distance from each word pair to the cluster center
#             for pair in word_pairs:
#                 word_vector = get_word_vector_from_pair(pair, glove_model)
                
#                 # Calculate cosine similarity between word pair vector and cluster center
#                 similarity = cosine_similarity([word_vector], [center])[0][0]
                
#                 # Cosine similarity is higher when vectors are closer, so we use (1 - similarity) to get a distance measure
#                 distance = 1 - similarity
#                 distances.append((pair, distance))
            
#             # Sort the word pairs by distance to the center (ascending order of distance)
#             distances.sort(key=lambda x: x[1])
            
#             # Pick the top `num_pairs` closest word pairs
#             cluster_selected_pairs[label] = [pair for pair, _ in distances[:num_pairs]]

#         # Add the selected pairs for the current category
#         selected_pairs[category] = cluster_selected_pairs

#     return selected_pairs


# Calculated by Euclenian distance
# def EUC_pick_closest_word_pairs_to_centers(clustered_data, glove_model, num_pairs=3):
    selected_pairs = {}

    # Set to track globally selected word pairs to ensure uniqueness
    global_selected_pairs = set()

    # Iterate through each category in clustered_data
    for category, data in clustered_data.items():
        print(f'Calculating category by EUC: {category}')
        
        # Extract the values for the current category
        cluster_results = data["clusters"]
        kmeans = data["kmeans"]

        # Get the cluster centers from kmeans
        centers = kmeans.cluster_centers_

        # Dictionary to store the closest word pairs for each cluster
        cluster_selected_pairs = {}

        # Iterate through each cluster
        for label, word_pairs in cluster_results.items():
            # Get the center of the current cluster
            center = centers[label]

            # List to store the distance of each pair from the center
            distances = []

            # Calculate the distance from each word pair to the cluster center
            for pair in word_pairs:
                # Calculate the word vector
                word_vector = get_word_vector_from_pair(pair, glove_model)

                # Calculate Euclidean distance between word pair vector and cluster center
                distance = np.linalg.norm(word_vector - center)  # Euclidean distance

                # Append the pair and its distance
                distances.append((pair, distance))

            # Sort the word pairs by distance to the center (ascending order of distance)
            distances.sort(key=lambda x: x[1])

            # Pick the top `num_pairs` closest unique word pairs
            unique_pairs = []
            for pair, _ in distances:
                if pair not in global_selected_pairs:
                    unique_pairs.append(pair)
                    global_selected_pairs.add(pair)

                # Stop if we have enough unique pairs
                if len(unique_pairs) >= num_pairs:
                    break

            cluster_selected_pairs[label] = unique_pairs

        # Add the selected pairs for the current category
        selected_pairs[category] = cluster_selected_pairs

    return selected_pairs

def calculate_distances(clustered_data, glove_model, num_pairs, distance_metric):
    selected_pairs = {}
    global_selected_pairs = set()

    for category, data in clustered_data.items():
        print(f"Calculating category: {category} using {distance_metric.__name__}")
        cluster_results = data["clusters"]
        kmeans = data["kmeans"]
        centers = kmeans.cluster_centers_
        cluster_selected_pairs = {}

        for label, word_pairs in cluster_results.items():
            center = centers[label]
            distances = []

            for pair in word_pairs:
                word_vector = get_word_vector_from_pair(pair, glove_model)
                distance = distance_metric(word_vector, center)
                distances.append((pair, distance))

            distances.sort(key=lambda x: x[1])
            unique_pairs = []
            for pair, _ in distances:
                if pair not in global_selected_pairs:
                    unique_pairs.append(pair)
                    global_selected_pairs.add(pair)
                if len(unique_pairs) >= num_pairs:
                    break
            cluster_selected_pairs[label] = unique_pairs

        selected_pairs[category] = cluster_selected_pairs
    return selected_pairs

def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

def cosine_distance(vec1, vec2):
    similarity = cosine_similarity([vec1], [vec2])[0][0]
    return 1 - similarity

def parallel_pick_closest_word_pairs(clustered_data, glove_model, num_pairs=3):
    with ThreadPoolExecutor() as executor:
        euc_future = executor.submit(
            calculate_distances, clustered_data, glove_model, num_pairs, euclidean_distance
        )
        cos_future = executor.submit(
            calculate_distances, clustered_data, glove_model, num_pairs, cosine_distance
        )
        euc_result = euc_future.result()
        cos_result = cos_future.result()

    return euc_result, cos_result

# def EUC_pick_closest_word_pairs_to_centers(clustered_data, glove_model, num_pairs=3):
#     selected_pairs = {}

#     # Iterate through each category in clustered_data
#     for category, data in clustered_data.items():
#         print('Calculating category by EUC')
#         # Extract the values for the current category
#         cluster_results = data["clusters"]
#         kmeans = data["kmeans"]
#         word_pairs = data["pairs"]
#         # cleaned_tokens = data["words"] 

#         # Get the cluster centers from kmeans
#         centers = kmeans.cluster_centers_

#         # Dictionary to store the closest word pairs for each cluster
#         cluster_selected_pairs = {}

#         # Iterate through each cluster
#         for label, word_pairs in cluster_results.items():
#             # Get the center of the current cluster
#             center = centers[label]
            
#             # List to store the distance of each pair from the center
#             distances = []
            
#             # Calculate the distance from each word pair to the cluster center
#             for pair in word_pairs:
#                 word_vector = get_word_vector_from_pair(pair, glove_model)
                
#                 # Calculate Euclidean distance between word pair vector and cluster center
#                 distance = np.linalg.norm(word_vector - center)  # Euclidean distance
#                 distances.append((pair, distance))
            
#             # Sort the word pairs by distance to the center (ascending order of distance)
#             distances.sort(key=lambda x: x[1])
            
#             # Pick the top `num_pairs` closest word pairs
#             cluster_selected_pairs[label] = [pair for pair, _ in distances[:num_pairs]]

#         # Add the selected pairs for the current category
#         selected_pairs[category] = cluster_selected_pairs

#     return selected_pairs

#  -----------------FILTER---------------------
# Removing stopper. Can add stoppers in all_stopwords_gensim
def Filtering(words_raw):
    all_stopwords_gensim = STOPWORDS.intersection(set(["."]))
    filtered_1 = []
    tokens_without_sw = [word for word in words_raw if not word in all_stopwords_gensim]
    filtered_1 += tokens_without_sw
    return filtered_1



#  Lemmatizing: combining words variation to be analyzed as single term
def lemmatizer(filtered_tokens):
    lemmatizer = WordNetLemmatizer()

    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    return lemmatized_tokens


# takes in string sentence of lemmatized words
#  Return sentiment score (1 = positive meaning, 0 = negative meaning)
def get_sentiment(text):
    # initiate the analyzer module
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    sentiment = int(1) if scores['compound'] > 0 else int(0)

    return sentiment


#-----------------------Word expansion-------------
def Expansion(sentence, glove_model):
    versions = expand_contractions(sentence)
    return "".join(versions)
    # For more computational power
    # if len(versions) == 1:
    #     return "".join(versions)
    # # Compare sentences and get similarity scores
    # similarities = compare_sentences(versions, glove_model)
    
    # sorteds = sorted(similarities, key=lambda x: x[2], reverse=True)
    # # # Print the sentence with the highest similarity
    # # most_similar_sentence_index = sentence_similarity_scores[0][0]
    # return "".join(versions[sorteds[0][0]])



def testing():
    # ---------------------------------Testing Zone-------------------------
    # # simulation of gamelist's structure
    # my_dict = {
    #     "game1": {"num" : 1},
    #     "game2": {"num": 2, 
    #                 "response" : ["I've put 788 hours into 7 Days to Die, and I've seen it evolve through several updates that have significantly changed the game. I played a lot before the Unity update, which is where most of my hours come from. Initially, the game struggled a bit after that update, but they eventually worked out the kinks.", 
    #                              "It's a great game for a beta version of a redo of the original game i highly recommend for new players and experienced survival horror tower defense/crafting games. It is a great time passer plus alot of fun, there are alot of small bugs here and there when driving vehicles but other than that i can't wait for whats to come when game is offically out of beta version.",
    #                              "I gotta tell you something. It was perfect. Perfect. Everything"]},
    #     "game3": {"num": 1, 
    #                 "response" : ['stuff 1', 'stiff 2']}
    # }

    # print(Mapping(my_dict, 'game2'))

    # text = "Nick likes to play football, however he is not too fond of tennis."
    # text_tokens = text.split()
    # print(text_tokens)
    # tokens_without_sw = [word for word in text_tokens if not word in all_stopwords_gensim]

    # print(tokens_without_sw)

    # # Add a new dictionary to "person1"
    # my_dict["game1"]["response"] = ["city", "New York", "zip", "10001"]
    # my_dict["game1"]["response"].append("now in charge") 
    # print(my_dict)
    # my_dict["game1"]["num"] += 2
    # print(my_dict)



    # for k, v in list(my_dict.items()):
    #     if v["num"] < 3:
    #         del my_dict[k]
            
    #     print("key is: " , k)
    #     print("value is: %s" % v)
    #     print("###########################")

    # print(my_dict)

    #-------------------------Testing zone 2--------------------
    # with open("BlackF.json", "r", encoding="utf-8") as file:
    #         Pirate1 = js.load(file)
    # data_block = Pirate1.pop("Black Flag")
    data_block = {
        "num": 39,
        "response": [
        "The best Assassin's creed game, thanks to it having lots of gameplay that has nothing to do with Assassin't creed.",
                "I love this game it's great. But this stupid game keeps lagging. I have turned nearly all of the graphics as low as possible and it still runs like a 90 year old grandma, IT DOESN'T. Not worth the money despite the great game play. I'm highly disappointed. I liked it until it wouldn't stop freezing and lagging.",
                "This game is ♥♥♥♥ing broken, I gave it every chance.... Console.... Keyboard....    I made it to the same point and then it gets into a pause loop.   ♥♥♥♥ YOU anyone that says they had a smooth run.   I'm happy to show you the ♥♥♥♥♥iness that I experienced if you like.  FIRST ♥♥♥♥ING HAND.  This game had potential....  now anyone that asks me will be met with \"it's♥♥♥♥♥♥ don't waste your money\"",
                "Edward James Kenway, Captain Kenway, great assassin.",
                "Assassin s Creed IV Black Flag is one of the best adventure free open world games that   I ever seen !\nStory line of the game is based on fighting in Caribean sea for freedom of your pirates and smashing the Britains and Spanish troops in the ,,New World ,,!Story is awesome !One of the best campaigns ever !Multiplayer is great !Combat and stealth fighting are also well done !There is a lot of intresting side missions and quests !\nI am big AC fan and for me this is best AC ever !",
                "Sucky. Combat is weird and not intuitive (space sometimes, E sometimes, whatever else sometimes)  and hiding is moronic. A stealth game where I can't crouch behind something? I've tried other Asassin's Creeed games and liked some a little and dsliked others intensely. Put this on the dislike list.",
                "why I cant play this game he requested activation code and I did it but next time I wanted to play this game again he wanted code again but this code didnt work. HELP",
                "I reallyt wanted to like this game, but it actually makes me seasick. No joke :(",
                "Highly recommend!!!!!!!!!!!!!!!!  This game is wonderful the controls are great and the story is so rich. Finishing this game made me just feel disheartened to play any other game because no other game could live up to this one so take my advice and draw it out and savour it as long as you can!",
                "One of the best games ive ever played and the best Assassin's Creed Game ever. Not anywhere else have I seen a company nail the pirate theme so well. I actually dont know why more companies dont try the idea. Moral of the story, If you like naval pirate combat, only to then dock your boat and still hold the bad♥♥♥♥♥stigma of an Assassin's Creed assassin, this game is for you.",
                "Reminds me of college.",
                "It's Ubisoft but I'll let that slide for now.\nPirate dude fights other pirate dudes\nStory - Meh.jpg\nControls - 6/10 - Combat is bad\nParkour - 8/10 - Few fixes could bump this up\nShips - 7/10\nGraphics - 9/10\n30 fps - gg reeeee (you can fix it but meh)\nYarr harr diddle dee dee, you are a pirate",
                "Was never that interested in AC as a series until AC3 took the series to more modern times. Something about running around during the revolution seemed fun. The naval missions felt out of place, though, and the story felt lacking.\nBlack Flag took those out-of-place naval missions and centered most of the new gameplay around sailing the high seas, upgrading your ship, and occasionally making landfall for various story reasons. It gives the game a vast, organic feel that I thoroughly enjoyed.\nI bought the game a number of years ago during a summer sale, but I still come back to it on occasion for some swashbuckling. Definitely recommend it.",
                "One of the best Assassin's Creed games with a refreshing pirate setting and mostly cohesive game elements that don't feel stitched together like they did in AC3.",
                "Do not by this game if you don't want to be annoyed, i have not played the game due to ♥♥♥♥♥♥♥t issues with uplay a secondary server you have to connect to in order to play this ♥♥♥♥ing game, like really you can't even just buy and play thru steam you have to jump thru extra hoops and download extra♥♥♥♥♥♥and set up an account with uplay just to play this ♥♥♥♥ing thing. talk about ♥♥♥♥♥♥♥t... Do not buy i'll be refunding my copy.",
                "♥♥♥♥ Ubisoft and ♥♥♥♥ any game they sell on steam!  Can't even install it because of Uplay go ♥♥♥♥ yourselves",
                "Still the best Assassins release!"
        ]
    }
    filtered_data = filter_short_responses(data_block, 55)
    sentences = ParToSen(filtered_data['response'])
    # Load GloVe model (you can change the path to the GloVe file on your system)
    glove_model_path = "preTrainData/glove.6B.100d.txt"  # Adjust path accordingly
    glove_model = load_glove_model(glove_model_path)
    print('glove loaded')
    for i in range(len(sentences)):
        if has_contraction(sentences[i]):
            sentences[i] = Expansion(sentences[i], glove_model)
    print('Sentences expanded')        
    # Now we have sentences list        

    # Process sentences and cluster word pairs
    clustered_data = process_sentences_and_cluster(sentences, glove_model, num_clusters=2)

    # # Print clustered data
    # for key, data in clustered_data.items():
    #     print(f"\nKey: {key}")
    #     print(f"Combinations: {data['combinations']}")
    #     print(f"Clusters: {data['clusters']}")
    #     print(f"Words: {data['words']}")
    print('done clustering')
    print('computing by cosine similarity')
    # Now we call the pick_densest_word_pairs function
    densest_word_pairs = COS_pick_closest_word_pairs_to_centers(clustered_data, glove_model, num_pairs=3)

        # Print densest word pairs for each cluster
    for category, clusters in densest_word_pairs.items():
        print(f"Category: {category}")
        for cluster_label, pairs in clusters.items():
            print(f"  Cluster {cluster_label}:")
            for pair in pairs:
                print(f"    {pair}")
    print(' ')
    print('computing by euclidean distance')     
    closest_pairs = EUC_pick_closest_word_pairs_to_centers(clustered_data, glove_model, num_pairs=3)

    # Print the closest word pairs for each category and cluster
    for category, clusters in closest_pairs.items():
        print(f"Category: {category}")
        for cluster_label, pairs in clusters.items():
            print(f"  Cluster {cluster_label}:")
            for pair in pairs:
                print(f"    {pair}")       
    
    # clustered_results is the output of your clustering process
    # densest_pairs = pick_densest_word_pairs(clustered_data, num_pairs=3)

    # for category, pairs in densest_pairs.items():
    #     print(f"Category: {category}")
    #     for pair in pairs:
    #         print(pair)
    # print('sentences of Cosine')
    # for i in range(5):
    #     print(generate_sentences(random_word_pairs_with_same_sentiment(densest_word_pairs)))
    
    
    # print('sentences of Euclidean')
    # for i in range(5):
    #     print(generate_sentences(random_word_pairs_with_same_sentiment(closest_pairs)))
    # # Print the clusters for each POS combination
    # for pos_type, data in clustered_data.items():
    #     print(f"POS Type: {pos_type}")
    #     for cluster_id, words in data["clusters"].items():
    #         print(f"  Cluster {cluster_id}: {words}")
    # result = pairs_with_sentiment(data_block)
    # print(result)
