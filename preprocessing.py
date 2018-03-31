import nltk
import pandas as pd
import re, string 
import pickle
import numpy as np

class PreProcessing:
    stances_path = ''
    bodies_path = ''

    def __init__(self, stances_path, bodies_path):
        self.stances_path = stances_path
        self.bodies_path = bodies_path

    def load(self):
        self.stances = pd.read_csv(self.stances_path, encoding = 'utf-8')
        self.bodies = pd.read_csv(self.bodies_path, encoding = 'utf-8')

        # Rename columns
        self.bodies.columns = ["bodyId", "articleBody"] 
        self.stances.columns = ["headline", "bodyId", "stance"] 

    def remove_punctuation(self, sentence):
        punctuations = string.punctuation + '—' + '’' + '…' + '‘' + '–' + '”' + '“'
        regex = re.compile('[%s]' % re.escape(punctuations))
        return regex.sub('', sentence)

    def remove_new_line(self, sentence):
        return re.sub(r"\n", " ", sentence)

    # Replace 'll to will 're to are 'm to am
    def full_sentence(self, sentence):
        sentence = re.sub(r"i'm", "i am", sentence)
        sentence = re.sub(r"he's", "he is", sentence)
        sentence = re.sub(r"she's", "she is", sentence)
        sentence = re.sub(r"it's", "it is", sentence)
        sentence = re.sub(r"that's", "that is", sentence)
        sentence = re.sub(r"what's", "what is", sentence)
        sentence = re.sub(r"where's", "where is", sentence)
        sentence = re.sub(r"how's", "how is", sentence)
        sentence = re.sub(r"\'ll", " will", sentence)
        sentence = re.sub(r"\'ve", " have", sentence)
        sentence = re.sub(r"\'re", " are", sentence)
        sentence = re.sub(r"\'d", " would", sentence)
        sentence = re.sub(r"\'re", " are", sentence)
        sentence = re.sub(r"won't", "will not", sentence)
        sentence = re.sub(r"can't", "cannot", sentence)
        sentence = re.sub(r"n't", " not", sentence)
        sentence = re.sub(r"n'", "ng", sentence)
        sentence = re.sub(r"'bout", "about", sentence)
        sentence = re.sub(r"'til", "until", sentence)

        return sentence

    def clean(self):
        print("Cleaning data...")
        # Lower the cases
        self.stances.headline = self.stances.headline.str.lower()
        self.bodies.articleBody = self.bodies.articleBody.str.lower()

        # Remove the punctuation
        self.stances.headline = self.stances.headline.apply(self.remove_punctuation)
        self.bodies.articleBody = self.bodies.articleBody.apply(self.remove_punctuation)

        # Remove the new line
        self.stances.headline = self.stances.headline.apply(self.remove_new_line)
        self.bodies.articleBody = self.bodies.articleBody.apply(self.remove_new_line)

        # Change to full sentence
        self.stances.headline = self.stances.headline.apply(self.full_sentence)
        self.bodies.articleBody = self.bodies.articleBody.apply(self.full_sentence)
        print("Cleaning data done...")

    def load_glove(self):
        self.word_vec = {}
        print("Start loading word vector...")
        with open('glove.txt', encoding="utf-8") as glove:
            count=0
            for line in glove:
                temp = line.split()
                l = len(temp)
                self.word_vec[' '.join(temp[0:l-50])] = list(map(np.float,temp[l-50:l]))
                count = count + 1

        print("Finish loading word vector...")

    def process_data(self):
        # Convert body into hash table
        self.bodies_hash = {}
        self.bodies_hash = {self.bodies.iloc[i].bodyId: self.bodies.iloc[i].articleBody for i in range(0, len(self.bodies))}

        raw_training_set = []
        data_related = []
        data_stances = []
        
        for index, stance in self.stances.iterrows():
            # Result: [[headline, body, stance]]
            raw_training_set.append([stance.headline, self.bodies_hash[stance.bodyId], stance.stance])

        # Backup raw data
        with open("data.p", "wb") as embedding:
            pickle.dump(raw_training_set, embedding)

        with open("data_related.p", "wb") as embedded_data_related:
            embedded_data_stances = open("data_stances.p", "wb")
            embedded_data_stances_dev = open("data_stances.dev.p", "wb")
            embedded_data_related_dev = open("data_related.dev.p", "wb")

            for sample in raw_training_set:
                # Split into words
                title = sample[0].split()
                body = sample[1].split()
                stance = sample[2]

                title_vector = []
                body_vector = []
                
                # Convert each words into vector
                for t in title:
                    if t in self.word_vec:
                        word = self.word_vec[t]
                        title_vector.append(word)
                for b in body:
                    if b in self.word_vec:
                        word = self.word_vec[b]
                        body_vector.append(word)

                # Split data into two parts, related and unrelated
                if stance == "unrelated":
                    data_related.append([title_vector, body_vector, [1, 0]])
                else:
                    data_related.append([title_vector, body_vector, [0, 1]])
                    
                    if stance == "discuss":
                        stance = [1,0,0]
                    if stance == "agree":
                        stance = [0,1,0]
                    if stance == "disagree":
                        stance = [0,0,1]
                    data_stances.append([title_vector, body_vector, stance])

            data_related = np.array(data_related)
            data_stances = np.array(data_stances)
            
            # Shuffle dataset
            np.random.shuffle(data_related)
            np.random.shuffle(data_stances)

            pickle.dump(data_related[0:3999, :], embedded_data_related_dev)
            pickle.dump(data_stances[0:3999, :], embedded_data_stances_dev)

            pickle.dump(data_related[4000:, :], embedded_data_related)
            pickle.dump(data_stances[4000:, :], embedded_data_stances)


pr = PreProcessing('./fnc-1/train_stances.csv', './fnc-1/train_bodies.csv')
pr.load()
pr.load_glove()
pr.clean()
pr.process_data()

