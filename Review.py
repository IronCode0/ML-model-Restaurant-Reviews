print( "Importing resources " )
from os import system
try:
    print( "pandas..." )
    import pandas as pd
except:
    print( "Failed to import 'pandas' \nprogram will exit" )
    system( "pause" )
    exit()

system( "cls" )
print( "Importing resources " )
from os import system
try:
    print( "re..." )
    import re
except:
    print( "Failed to import 're' \nprogram will exit" )
    system( "pause" )
    exit()

system( "cls" )
print( "Importing resources " )
from os import system
try:
    print( "nltk..." )
    import nltk
except:
    print( "Failed to import 'nltk' \nprogram will exit" )
    system( "pause" )
    exit()
    
system( "cls" )
file_src = "Restaurant_Reviews-211023-184653.tsv"
file_len = 1000
print( "Checking Review from file: " + file_src )


dataset = pd.read_csv(file_src,delimiter = "\t",quoting = 3)

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []

print( "...Filtering and arranging data" )
for i in range(0,file_len):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[: , -1].values

print( "...Training model" )
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)
#from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import accuracy_score
#cm = confusion_matrix(y_test, y_pred)
ac = str(accuracy_score(y_test, y_pred))
print( "------------RESULT------------" )
#print( "confusion matrix" )
#print( cm )
print( "\naccuracy score = " + ac)
system('pause')
exit()
