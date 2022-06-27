#Code created by Utkarsh Prajapati 

#Importing libraries
import pandas as pd
import numpy as np
import sklearn
import nltk
import re
import string
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score as a_s

#Downloading Stopwords and Punctuators
nltk.download("punkt")
nltk.download('stopwords')

#Initializing Stopword Set and Stemmer
sw=set(nltk.corpus.stopwords.words("english"))
stemmer=SnowballStemmer("english")

#Reading Data from CSV Dataset and Filtering it
data=pd.read_csv("tweets.csv")
data["label"]=data["class"].map({0:"Hate Speech", 1: "Offensive Speech", 2: "No Hate and Offensive Speech"})
data=data[["tweet","label"]]

#Performing data Cleansing
def cleanse(t):
    t=str(t).lower()
    t=re.sub("[?.]","",t)
    t=re.sub("https?:\S+|www.\S+","",t)
    t=re.sub("<.?>+","",t)
    #Removing Puntuations
    t=re.sub('[%s]' % re.escape(string.punctuation),"",t)
    t=re.sub("\n","",t)
    t=re.sub("\w\d\w","",t)
    #Removing Stopwords
    t=[w for w in t.split(" ") if not w.lower() in sw]
    t=" ".join(t)
    #Using Snowball Stemming(Porter2)
    t=[stemmer.stem(w) for w in t.split(" ")]
    t=" ".join(t)
    return t
data["tweet"]=data["tweet"].apply(cleanse)

x=np.array(data["tweet"])
y=np.array(data["label"])
cv=CountVectorizer() #Unique Word Frequency Vectors are created
X=cv.fit_transform(x) #Sparse Matrix is Created

#Making training and Test Data of 70-30 ratio
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

#Using Decision Tree Classifier Model
model=DecisionTreeClassifier()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

#Evaluating Model Accuracy 
from sklearn.metrics import accuracy_score as a_s
print("Model Accuracy:- ",a_s(y_test,y_pred)*100,"%")

#Try it yourself
a="@kikiSTFU: I'll break that lil bitch neck nd won't even feel sorry bout it" #Change value of variable to what you want to try it out
a=cv.transform([a]).toarray()
print("Result:- ",model.predict(a)[0])
