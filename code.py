import numpy as np
import pandas as pd
df=pd.read_csv('spam_ham_dataset.csv')
df.head() #Looking at Top 5 records in the Data
df.info() #Understanding the datatypes of the columns in the dataframe
df.isna().sum() #Checking Whether the Data contains any Null Values

import seaborn as sns #A library that uses matplotlib underneath to plot graphs,used to visualize random distributions
sns.countplot(x="label",data=df) #Composition of Spam and Ham mails in the Data using simple CountPlot of Seaborn

print("Data Cleaning \n",df['text'][0],"\n") #Data Cleaning

!pip install contractions

import contractions
#contractions is the package in python used to expand the contractions in english language to their original form. Example: I'll to "I will"
from tqdm import tqdm
#tqdm package is used to track the progress of work. It displays the percentage of loop done.
import nltk
#nltk is a suite of libraries that are mainly used for dealing with problems related to Natural language processing.
import re
#re means regular expression
nltk.download('stopwords')
from nltk.corpus import stopwords
#downloading the stopwords of english language
stopwords=stopwords.words('english')
#Removing stopwords 'no','nor' and 'not' as these may add value to the text
stopwords.remove('no')
stopwords.remove('nor')
stopwords.remove('not')

processed_mails=[]
for i in tqdm(df['text'],desc='Mails Processing Progress '):
    #Regular expression that removes all the html tags present
    i=re.sub('(<[\w\s]*/?>)',"",i)
    #Expanding all the contractions present in the review to it's respective actual form
    i=contractions.fix(i)
    #Removing all the special characters from the review text
    i=re.sub('[^a-zA-Z0-9\s]+',"",i)
    #Removing all the digits present in the review text
    i=re.sub('\d+',"",i)
    #Making all the review text to be of lower case as well as removing the stopwords and words of length less than 3
    processed_mails.append(" ".join([j.lower() for j in i.split() if j not in stopwords and len(j)>=3]))

#Creating a new dataframe using the Processed Reviews
processed_df=pd.DataFrame({'text':processed_mails,'Spam/Ham':list(df['label_num'])})
processed_df.head()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download('stopwords')

df=pd.read_csv('spam_ham_dataset.csv')
print(df)
df.drop(columns='Unnamed: 0', axis=1, inplace=True)
print("After")
print(df,"\n")
print(df.isna().sum(),"\n")
#sns.countplot(df.label_num)
import string
#from nltk.corpus import stopwords

# remove stopword & punctuation
def text_preprocessing(message):
    stop_words = stopwords.words('english')
    
    punc = [token for token in message if token not in string.punctuation]
    punc = ''.join(punc)
    
    return ''.join([word for word in punc.split() if word .lower() not in stop_words])

df['clean_message'] = df.text.apply(text_preprocessing)
# print("Here",df['clean_message'])

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
#Convert a collection of raw documents to a matrix of TF-IDF(term frequency-inverse document frequency) features
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('spam_ham_dataset.csv')

print("\n"+'-'* 30)
print("\nshape of data:",data.shape) 
print("\n"+'-'* 30)
print("\nno dimensions of data:",data.ndim)
print("\n"+'-'* 30)
print("\nsize of data:",data.size)
print("\n"+'-'*30)
print("\nSum of all null values:\n",data.isnull().sum())
print("\n"+'-'*30)

#Data Visualization
# Vis = data['label_num'].value_counts()
# Vis.plot(kind="bar")
# plt.xticks(np.arange(2), ('Non spam', 'spam'),rotation=0)
# plt.show()

#label spam mail as 0;  ham mail as 1;

data.loc[data['label'] == 'spam', 'Category',] = 0
data.loc[data['label'] == 'ham', 'Category',] = 1

# separating the data as texts and label

X = data['text']

Y = data['label']

#Train_Test_Splitting

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=3)
#random state hyperparameter function controls the shuffling process(same train and test set for different executions).

y_train=[]
y_train_value=[]
y_test=[]
y_test_value=[]

for i in Y_train:
  y_train.append(i)
  if(i == "spam") : y_train_value.append(0)
  else : y_train_value.append(1)
df_y_train_value = pd.DataFrame(y_train_value)

for i in Y_test:
  y_test.append(i)
  if(i == "spam") : y_test_value.append(0)
  else : y_test_value.append(1)
df_y_test_value = pd.DataFrame(y_test_value)

#Future Extraction
#transform the text data to feature vectors that can be used as input to the Logistic regression

feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase='True')

X_train_transformed = feature_extraction.fit_transform(X_train)
X_test_transformed = feature_extraction.transform(X_test)

tfidf_tokens = feature_extraction.get_feature_names_out()
#use get_feature_names if that won't work
df_tfidfvect = pd.DataFrame(data = X_train_transformed.toarray(),columns = tfidf_tokens)
print("Tokens :")
print(tfidf_tokens)
print("\nTD-IDF Vectorizer :")
print(df_tfidfvect)
print("\nTD-IDF Vectorizer Shape :")
print(df_tfidfvect.shape)
print("\n")
print(X_train_transformed)
print(X_train_transformed.shape)
print("Next\n")
print(X_test_transformed)
print(X_test_transformed.shape)

#Logistic Regression
model = LogisticRegression()

# training the Logistic Regression model with the training data
model.fit(X_train_transformed, Y_train)

print("\nFor Test Set of 30% :")

# prediction on training data

prediction_on_training_data = model.predict(X_train_transformed)
#print(prediction_on_training_data)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
print('Accuracy on training data : ', accuracy_on_training_data *100)

# prediction on test data

prediction_on_test_data = model.predict(X_test_transformed)
#print(prediction_on_test_data)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print('Accuracy on test data : ', accuracy_on_test_data*100)

train_data_value=[]
for i in prediction_on_training_data:
  if(i == "spam") : train_data_value.append(0)
  else : train_data_value.append(1)
df_train_data_value = pd.DataFrame(train_data_value)

test_data_value=[]
for i in prediction_on_test_data:
  if(i == "spam") : test_data_value.append(0)
  else : test_data_value.append(1)
df_test_data_value = pd.DataFrame(test_data_value)

print("\nTrain Mean Square Error :")
print(mean_squared_error(df_y_train_value, df_train_data_value))

print("\nTest Mean Square Error :")
print(mean_squared_error(df_y_test_value, df_test_data_value))

print("\nConfusion matrix :")
print(confusion_matrix(y_test, prediction_on_test_data, labels=["spam", "ham"]))

tn, fp, fn, tp = confusion_matrix(y_test, prediction_on_test_data, labels=["spam", "ham"]).ravel()

print("\nIn order of tn, fp, fn, tp :")
print(tn, fp, fn, tp)

print("\nClassification report :")
print(classification_report(y_test, prediction_on_test_data, labels=["spam", "ham"]))

precision = tp/(tp+fp)
recall = tp/(tp+fn)

print("\nPrecision :")
print(precision)

print("\nRecall :")
print(recall,"\n")

sns.heatmap(confusion_matrix(y_test, prediction_on_test_data, labels=["spam", "ham"]))