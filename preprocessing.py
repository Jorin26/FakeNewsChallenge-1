import nltk
import pandas as pd
import re, string 

class PreProcessing:
    stances_path = ''
    bodies_path = ''

    def __init__(self, stances_path, bodies_path):
        self.stances_path = stances_path
        self.bodies_path = bodies_path

    def load(self):
        self.stances = pd.read_csv(self.stances_path, encoding = 'utf-8')
        self.bodies = pd.read_csv(self.bodies_path, encoding = 'utf-8')

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
        # Lower the cases
        self.stances.Headline = self.stances.Headline.str.lower()
        self.bodies.articleBody = self.bodies.articleBody.str.lower()

        # Remove the punctuation
        self.stances.Headline = self.stances.Headline.apply(self.remove_punctuation)
        self.bodies.articleBody = self.bodies.articleBody.apply(self.remove_punctuation)

        # Remove the new line
        self.stances.Headline = self.stances.Headline.apply(self.remove_new_line)
        self.bodies.articleBody = self.bodies.articleBody.apply(self.remove_new_line)

        # Change to full sentence
        self.stances.Headline = self.stances.Headline.apply(self.full_sentence)
        self.bodies.articleBody = self.bodies.articleBody.apply(self.full_sentence)

        return self.stances, self.bodies


