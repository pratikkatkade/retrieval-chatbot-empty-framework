from user_functions import preprocess, compare_overlap, pos_tag, extract_nouns, compute_similarity
from collections import Counter
import spacy
word2vec = spacy.load('en')

blank_spot = ""

response_a = ""
response_b = ""
response_c = ""

responses= [response_a, response_b, response_c]

class ChatBot:
    def find_intent_match(self, responses, user_message):
        bow_user_message = Counter(preprocess(user_message))
        processed_responses = [Counter(preprocess(response)) for response in responses]
        similarity_list = [compare_overlap(doc, bow_user_message) for doc in processed_responses]
        response_index = similarity_list.index(max(similarity_list))
        return responses[response_index]

    def find_entities(self, user_message):
        tagged_user_message = pos_tag(preprocess(user_message))
        message_nouns = extract_nouns(tagged_user_message)
        
        tokens = word2vec(" ".join(message_nouns))
        category = word2vec(blank_spot)
        word2vec_result = compute_similarity(tokens, category)
        word2vec_result.sort(key=lambda x: x[2])
        return word2vec_result[-1][0]

    def respond(self,user_message):
        best_response = self.find_intent_match(responses, user_message)
        entity = self.find_entities(user_message)
        print(best_response.format(entity))
        print("I hope I was able to help. See ya around!")
        return True
    
    def chat(self):
        user_message = input("Hey! I'm a bot. Ask me your questions! ")
        self.respond(user_message)

ChatBot = ChatBot()
ChatBot.chat()

    