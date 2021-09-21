import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix , classification_report
df = pd.read_csv('SMSSpamCollection' , sep = '\t' , names = ['labels','messages'])

stemmer = PorterStemmer()
bow = []
for i in range(0, len(df)):
    new_sent = re.sub('[^a-zA-Z]','',df['messages'][i])
    new_sent = new_sent.lower()
    new_sent = new_sent.split()
    new_sent = [stemmer.stem(word) for word in new_sent if word not in set(stopwords.words('english'))]
    new_sent = ''.join(new_sent)
    bow.append(new_sent)

bag_of_words = CountVectorizer(max_features=5000)
X = bag_of_words.fit_transform(bow).toarray()
X = pd.DataFrame(X)

y = pd.get_dummies(df['labels'])
y = y.iloc[:,1].values

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.3)

imb = RandomUnderSampler()
X_train , y_train = imb.fit_resample(X_train , y_train)
X_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train)

bayes = MultinomialNB()
bayes.fit(X_train , y_train)
predictions = bayes.predict(X_test)

acc_1 = confusion_matrix(y_test , predictions)
acc_2 = classification_report(y_test , predictions)
print(acc_1)
print(acc_2)

