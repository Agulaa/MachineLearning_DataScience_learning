import random, nltk

nltk.download('names') #lista angielski męskich i żeńskich imion

male_names = nltk.corpus.names.words('male.txt')
female_names = nltk.corpus.names.words('female.txt')
print(male_names[:5])

labeled_names = [(name, 'male') for name in male_names ]
labeled_names = [(name, 'female') for name in female_names ]

random.seed(12347)
random.shuffle(labeled_names)
print(labeled_names[:5])

def get_features_names(name):
    return {
        'first_letter':name[0],
        'last_but_one_letter':name[-2],
        'last_letter':name[-1],
        'length':len(name),
    }


features = [(get_features_names(name), gender) for (name,gender) in labeled_names]

train_set, test_set = features[1000:], features[:1000]


bayes_classifier = nltk.NaiveBayesClassifier.train(train_set) #wytrenowanie klasyfikatora na zbiorze uczącym

print(bayes_classifier.classify(get_features_names('Alice'))) #sprawdzenie poprawności rozpoznawania płci

print(nltk.classify.accuracy(bayes_classifier, test_set)) # dpkładność klasyfiakra

bayes_classifier.show_most_informative_features(10)