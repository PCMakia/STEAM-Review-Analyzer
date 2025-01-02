import matplotlib.pyplot as plt
import ast
import os
import json as js
import processClustering as pc
import heapq
import random

templates = {
    "1": [
        "The {noun} {verb} {adj} {adv}.",
        "A {noun} {verb} {adj} {adv}.",
        "The {noun} is {adj} and {verb} {adv}.",
        "The {noun} {verb} {adj} every {adv}.",
        "The {noun} {verb} {adj} with great {adv}.",
        "The {noun} {verb} {adj} effortlessly {adv}.",
        "The {noun} {verb} {adj} with {adv} precision.",
        "A {noun} always {verb} {adj} {adv}.",
        "The {noun} {verb} {adj} like a {adv} champion.",
        "The {noun} {verb} {adj} and looks {adv}.",
        "The {noun} {verb} {adj} with a {adv} attitude.",
        "A {noun} {verb} {adj} because of its {adv} nature."
    ],
    "0": [
        "The {noun} {verb} {adj} {adv}.",
        "A {noun} {verb} {adj} {adv} with no {adv}.",
        "The {noun} is {adj} but doesn't {verb} {adv}.",
        "The {noun} {verb} {adj} even though it is {adv}.",
        "The {noun} {verb} {adj} poorly {adv}.",
        "The {noun} {verb} {adj} with {adv} difficulty.",
        "A {noun} {verb} {adj} despite being {adv}.",
        "The {noun} {verb} {adj} and struggles {adv}.",
        "The {noun} {verb} {adj} but never {adv}.",
        "A {noun} {verb} {adj} though it’s {adv} weak.",
        "The {noun} {verb} {adj} but is still {adv}.",
        "The {noun} {verb} {adj} badly {adv}."
    ]
}
#---------------------Sentence making---------------------------
# Getting tuple of 4 words with same sentiment score
def random_word_pairs_with_same_sentiment(selected_pairs, num_words=4):
    # List to store the selected word pairs
    selected_word_pairs = {
        "noun": None,
        "verb": None,
        "adj": None,
        "adv": None
    }
    
    # Get a list of categories
    categories = list(selected_pairs.keys())
    
    # Randomly shuffle the categories to ensure diversity
    random.shuffle(categories)
    
    # Keep track of the sentiment to ensure they are the same for all selected pairs
    selected_sentiment = None
    
    # Randomly grab one word pair from each distinct category
    for category in categories:
        # Get the word pairs for the current category
        word_pairs = selected_pairs[category]
        
        # Shuffle word pairs to randomize selection
        random.shuffle(word_pairs)
        
        # Find the first word pair that matches the desired sentiment
        for pair in word_pairs:
            sentiment = pair[0]  # The sentiment score is stored as the first item in the tuple
            if selected_sentiment is None:
                # Set the sentiment score for the first word pair selected
                selected_sentiment = sentiment
            elif selected_sentiment != sentiment:
                # Skip this word pair if its sentiment doesn't match the selected sentiment
                continue
            
            # Determine the type of word pair and assign the words to the correct category
            word1, word2 = pair[1], pair[2]
            
            if category == "noun-adjective":
                selected_word_pairs["noun"] = word1
                selected_word_pairs["adj"] = word2
            elif category == "noun-verb":
                selected_word_pairs["noun"] = word1
                selected_word_pairs["verb"] = word2
            elif category == "verb-adjective":
                selected_word_pairs["verb"] = word1
                selected_word_pairs["adj"] = word2
            elif category == "noun-adverb":
                selected_word_pairs["noun"] = word1
                selected_word_pairs["adv"] = word2
            elif category == "verb-adverb":
                selected_word_pairs["verb"] = word1
                selected_word_pairs["adv"] = word2
            elif category == "adjective-adverb":
                selected_word_pairs["adj"] = word1
                selected_word_pairs["adv"] = word2

            # Stop once we've selected one pair from each category
            if all(value is not None for value in selected_word_pairs.values()):
                break
        
        # Stop once we've selected enough word pairs
        if all(value is not None for value in selected_word_pairs.values()):
            break
    
    # If we have selected one word from each category, return the tuple
    if all(value is not None for value in selected_word_pairs.values()):
        # Return the sentiment score followed by noun, verb, adj, and adv
        return (selected_sentiment, selected_word_pairs["noun"], selected_word_pairs["verb"],
                selected_word_pairs["adj"], selected_word_pairs["adv"])
    else:
        return None  # Return None if we couldn't find enough valid word pairs with the same sentiment

# Pseudo sentence generation
def generate_sentences(category, noun, verb, adj, adv):
    if category not in templates:
        raise ValueError("Invalid category. Use '1' for positive or '0' for negative.")
    
    # Get 4 random templates from the selected category
    selected_templates = random.sample(templates[category], 4)
    
    # Generate the formatted sentences
    sentences = [f"{template.format(noun=noun, verb=verb, adj=adj, adv=adv)}" for template in selected_templates]
    
    return sentences

# takes in dictionary and threshold int. 
# Return dictionary without counting lower than threshold
def filtering(dict, threshold):
    for k, v in list(dict.items()):
        if v["num"] < threshold:
            del dict[k]
    return dict   

#  Takes in dictionary and a number (optional), return number of games id with highest review count
def highest_games(dict, num=10):
    all_games = {}
    for k, v in list(dict.items()):
        all_games[k] = v['num']
    return heapq.nlargest(num, all_games, key=all_games.get)[:num]

# # Get the game id with highest review
# def highest_review(dict):
#     highest = ""
#     for k, v in list(dict.items()):
#         if v["num"] > highest["num"] | highest ==  "":
#             highest = k
#     return highest       


def most_frequent_word(dict):
	all_games = [dict.items()]
	reviews_counts = [number for pid in dict for number in pid if number == "num"]
	return heapq.nlargest(1, all_games, key=reviews_counts.get)[0]

def read_data_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        output = []
        # Read the file content
        for line in file:
            # content = line.read()
            
            try:
                # Use ast.literal_eval to safely parse the content
                data = ast.literal_eval(line)
                output.append( data )
                
            except (SyntaxError, ValueError) as e:
                print(f"Error parsing the file: {e}")
                return None
    return output
# Function to extract matching titles based on ID comparison
def get_matching_titles(data, id_list):
    if not data:
        return []
    
    # Extract titles for matching IDs
    matching_titles = [
        item[u'title'] for item in data if u'id' in item and item[u'id'] in id_list
    ]
    return matching_titles


# Define the file path relative to the script location
def get_file_path(folder, fileN):
    # Get the directory of the current script
    current_dir = os.path.dirname(__file__)
    # Construct the path to the JSON file
    file_path = os.path.join(current_dir, folder, fileN)
    return file_path
# update the id by title
def replace_ids_with_titles(games_dict, data):
    # Extract game IDs from the dictionary keys, ensuring they are integers or strings as per the data structure
    id_list = list(games_dict.keys())
    
    # Create a mapping of ID to title from the data, ensuring proper matching of ID types
    id_to_title = {str(item['id']): item['title'] for item in data if 'id' in item and 'title' in item}
    
    # Replace game IDs with their titles in the dictionary, ensuring the order of IDs in games_dict
    updated_dict = {id_to_title.get(game_id, game_id): games_dict[game_id] for game_id in id_list}
    
    return updated_dict
def main():
    gamelist = {}
    # open the json object
 
    # Version 2

    # with open("./raw_steam_new.json") as f:
    #     for line in f:
    #         try:
    #             rec = ast.literal_eval(line)

    #             # cr is the current json file we are in
    #             # cr = js.loads(line)
    #             # rev: the review of user
    #             rev = rec["text"]
    #             pid = rec["product_id"]
    #             # Set a game list by dictionary (add new game and increase review"s count on existing game)
    #             if pid not in gamelist.keys():       
    #                 gamelist[str(pid)] = {"num" : 1}
    #                 gamelist[str(pid)]["response"] = [str(rev)]
    #             else:    
    #                 gamelist[str(pid)]["num"] +=1
    #                 gamelist[str(pid)]["response"].append(str(rev)) 
    #         except js.decoder.JSONDecodeError as e:
    #             json_decode += 1
    #             print(e)
    #             print("failed")
            #     pass
    # # Saved for later
    # with open("sample1.json") as f:
    #     try:
    #         gamelist = js.load(f)
    #         #     listgames =  ast.literal_eval(line)
    #         gamelist = filtering(gamelist, 10000)
    #         highest = highest_games(gamelist)
    #         with open("sample2.json", "w") as save:
    #              js.dump(gamelist, save)
    #         print(highest)
    #     except js.decoder.JSONDecodeError as e:
    # #             json_decode += 1
    # #             print(e)
    #             print("failed unloaded data")


    with open("sample2.json") as f:
        try:
            gamelist = js.load(f)
            #     listgames =  ast.literal_eval(line)
            highest = highest_games(gamelist)
            print(highest)
        except js.decoder.JSONDecodeError as e:
    #             json_decode += 1
    #             print(e)
                print("failed unloaded data")

    with open("gameinfo/steam_games.json") as file:
         for line in file:
            try:
                rec = ast.literal_eval(line)
                for i in range(len(highest)):
                    if rec["id"] == highest[i]:
                         highest[i] = rec['title']
            except js.decoder.JSONDecodeError as e:
    #             json_decode += 1
    #             print(e)
                print("failed name extraction")
    print(highest)
    # # Filtering the low review games 
    # filtering(gamelist, 10000)
    # # game(s) with highest reviews number
    # # test = highest_games(gamelist, 1)
    # listGames = highest_games(gamelist)
    
    # # map(mp.Mapping,listGames)
    # for game in listGames:
    #     processed = mp.Mapping(gamelist, game)
    #     output = Sim.most_frequent_word(processed[1])
    #     print(f"Users most often describe the playing experience of this game as {output['adjs']}.")
    #     print(f"Users most often describe this game as a {output['nouns']} type game.")
    #     print(f"Users most often {output['advs']} recommend this game to other players.")
    # #  Processing
    # # mapReduce the reviews. Should return list of useful verbs and adj, adv
    # # processed = mp.Mapping(gamelist, test)

    # # # Similar words (verb, adj, adv) are removed, leaving 1 word in each category 
    # # output = Sim.most_frequent_word(processed[1])
    # # # Adding those word into pre-written sentences
    # # # print("value is: %s, and the other is and that is it!" % v) # example of sentence with blank space
    
    # # # F Strings
    # # print(f"Users most often describe the playing experience of this game as {output['adjs']}.")
    # # print(f"Users most often describe this game as a {output['nouns']} type game.")
    # # print(f"Users most often {output['advs']} recommend this game to other players.")
    #     print(f"The {output['nouns']} for this game is {output['advs']} atmospheric, setting the tone for each level and creating a sense of immersion for the player.")
    # # print(f"Users who not enjoy the game describe it as {[output'adjective']}.")





#  Graphing code
    # sorted_game = dict(sorted(gamelist.items(), key=lambda item: item[1], reverse=True))        

    # plt.figure(figsize=(15,5))
    # # plt.bar(gamelist.keys(), gamelist.values(),edgecolor="black", width=1, color="g")
    # # # plt.hist(gamelist.values(),bins= 1000,edgecolor="black", width=.8, color="g")
    # # plt.title("Distribution of reviews among games")
    # # plt.xlabel("Game's title")
    # # plt.ylabel('number of reviews')
    # # plt.show()

    # plt.bar(sorted_game.keys(), sorted_game.values(),edgecolor="black", width=1, color="g")
    # # plt.hist(gamelist.values(),bins= 1000,edgecolor="black", width=.8, color="g")
    # plt.title("Distribution of reviews among games")
    # plt.xlabel("Game's title")
    # plt.ylabel('number of reviews')
    # # plt.show()

    # plt.bar(hist_game.keys(), hist_game.values(),edgecolor="black", width=1, color="b")
    # plt.title("Distribution of reviews among games by order")
    # plt.xlabel("Game's title")
    # plt.ylabel('number of reviews')
    # plt.show()

def main2():
    file_path = get_file_path('gameinfo','steam_games.json')
    # info_path = "gameinfo/steam_games.json" 

    with open("sample2.json") as f:
    
        gamelist = js.load(f)
        #     listgames =  ast.literal_eval(line)
        
    

    # Read and parse the data
    datainfo = read_data_from_file(file_path)

    GameNameList = replace_ids_with_titles(gamelist, datainfo)
    print(' ')
    highest = highest_games(GameNameList)
    print(highest)
    # Load GloVe model (you can change the path to the GloVe file on your system)
    glove_model_path = "preTrainData/glove.6B.100d.txt"  
    glove_model = pc.load_glove_model(glove_model_path)
    print('glove loaded')
    
    for game in highest:
        print(f'Currently: {game}')
        pointer = GameNameList[game]

        filtered_data = pc.filter_short_responses(pointer, 55)
        sentences = pc.ParToSen(filtered_data['response'])
        
        
        for i in range(len(sentences)):
            if pc.has_contraction(sentences[i]):
                sentences[i] = pc.Expansion(sentences[i], glove_model)
        print('Sentences expanded')        
        # Now we have sentences list        

        # Process sentences and cluster word pairs
        clustered_data = pc.process_sentences_and_cluster(sentences, glove_model, num_clusters=2)

        
        print('done clustering')
        print('computing by cosine similarity')

        euc_result, cos_result = pc.parallel_pick_closest_word_pairs(clustered_data, glove_model, num_pairs=8)

        with open('data.txt', 'a', encoding='utf-8') as file:
            for result, metric in zip([euc_result, cos_result], ['Euclidean Distance', 'Cosine Similarity']):
                file.write(f'Below are pairs computed using {metric} for {game}\n')
                file.write('\n')
                for category, clusters in result.items():
                    file.write(f"Category: {category}\n")
                    for cluster_label, pairs in clusters.items():
                        file.write(f"  Cluster {cluster_label}:\n")
                        for pair in pairs:
                            file.write(f"    {pair}\n")
            file.write('---------------------------------------------------------------------\n')                
    # Now we call the pick_densest_word_pairs function
    # densest_word_pairs = pc.COS_pick_closest_word_pairs_to_centers(clustered_data, glove_model, num_pairs=8)
    # with open('data.txt', 'a') as file:
    #     file.write(f'Below is pairs of game: {game}, computing by Cosine Similarity\n')
    #     file.write(f'\n')
    #     # Print densest word pairs for each cluster
    #     for category, clusters in densest_word_pairs.items():
    #         file.write(f"Category: {category}\n")
    #         for cluster_label, pairs in clusters.items():
    #             file.write(f"  Cluster {cluster_label}:\n")
    #             for pair in pairs:
    #                 file.write(f"    {pair}\n")
    #     file.write('\n')
    #     print(f'Currently: {game} done COS')
    #     file.write('computing by euclidean distance\n')     
    # closest_pairs = pc.EUC_pick_closest_word_pairs_to_centers(clustered_data, glove_model, num_pairs=8)

    # # Print the closest word pairs for each category and cluster
    # with open('data.txt', 'a') as file:
    #     file.write(f'Continue Computing {game} by Euclidean distance\n')
    #     file.write(f'\n')
    #     # Print densest word pairs for each cluster
    #     for category, clusters in closest_pairs.items():
    #         file.write(f"Category: {category}\n")
    #         for cluster_label, pairs in clusters.items():
    #             file.write(f"  Cluster {cluster_label}:\n")
    #             for pair in pairs:
    #                 file.write(f"    {pair}\n")
    #     print(f'Currently: {game} done EUC')


    # Output the results
    # print("Matching Titles:", titles)
    # #         Skyrim   Bayonetta  Black_Flag
    # monid = ['489830', '460790', '242050']
    # pirate = gamelist.pop('242050')
    # Pirate = {
    # "Black Flag": pirate
    # }
    # # Pirate["Black Flag"] = Pirate.pop("242050")
    # print (Pirate["Black Flag"]["num"])

    # The test data of AC:BF
    # with open("BlackF.json", "r", encoding="utf-8") as file:
    #     Pirate1 = js.load(file)
    # print(Pirate1['Black Flag']['num'])    
    # testgame = get_matching_titles(datainfo, monid)
    # print("Testing Title:", testgame)
main2()