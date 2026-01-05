
#Task 3
#Import dependencies
import pandas as pd
import requests
from textblob.classifiers import NaiveBayesClassifier
from sklearn.model_selection import KFold
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
import random

#Download the dataset
url = "https://storage.googleapis.com/dataset-uploader/bbc/bbc-text.csv"
df = pd.read_csv(url)

#Explore the dateset
#Print the total number of documents
print("Total documents:", len(df))

#Print the documents per class
print("\nDocuments per class:")
print(df['category'].value_counts())

#Print one sample document per class
print("\nSample document per class:\n")
for c in df['category'].unique():
  sample_text = df[df['category'] == c].iloc[0]['text']
  print(f"\n--- Category: {c} ---")
  #print the first 500 characters
  print(sample_text[:500], "...")
  print("-------------------------------")

#Train/Test 80/20 split
def train_test_split(data, test_size=0.2, seed=42):
  #Find fixed starting state
  random.seed(seed)
  #Make shallow copy of list
  data_copy = data.copy()
  #Randomly shuffle data in place so the train/test split is random, not ordered
  random.shuffle(data_copy)
  #Calculate how many should go in the training set (.8 or 80%)
  split_point = int(len(data_copy) * (1 - test_size))
  #Takes the first 80%
  train_data = data_copy[:split_point]
  #Takes the last 20%
  test_data = data_copy[split_point:]
  #Returns the two lists
  return train_data, test_data

#Prepare the text and label tuples
#Combines the data into tuples so that TextBlob can do its thing
data_tuples = list(zip(df['text'], df['category']))
#Perform the actual split from the data
train_data, test_data = train_test_split(data_tuples, test_size=0.2)

print("\nTraining samples:", len(train_data))
print("Test samples:", len(test_data))

#Train the Naive Bayes classifier
print("\nTraining Naive Bayes classifier... (this may take a few minutes)")
classifier = NaiveBayesClassifier(train_data)

#Evaluate the accuracy of the trained data
accuracy = classifier.accuracy(test_data)
print("\nModel accuracy on test set:", accuracy)

#2-fold cross-validation
print("\nPerforming 2-fold cross-validation...")
#Split the data into two and shuffle
#Set the seed at 42 so it has the same halves each tme to test for accuracy
kf = KFold(n_splits=2, shuffle=True, random_state=42)

fold_accuracies = []
#Stores the info from the tuples created earlier
text_label_pairs = data_tuples

#Return the index for each fold--folds twice
for train_index, test_index in kf.split(text_label_pairs):
  #Takes and trains the indexes provided by the KFold
  train_fold = [text_label_pairs[i] for i in train_index]
  test_fold = [text_label_pairs[i] for i in test_index]

  #Creates a new Naive Bayes classifier and train it on the folded trained set
  fold_cl = NaiveBayesClassifier(train_fold)
  #Tests the model on the test fold and gets the accuracy
  fold_acc = fold_cl.accuracy(test_fold)
  fold_accuracies.append(fold_acc)

print("Fold accuracies:", fold_accuracies)
print("Average 2-fold CV accuracy:", sum(fold_accuracies) / len(fold_accuracies))

#Task 4
#Import necessary dependencies
import nltk
nltk.download('punkt')
from textblob.classifiers import NaiveBayesClassifier

def naiveBayesClassifier(trainingSet, testSet):
    #Train the classifier
    classifier = NaiveBayesClassifier(trainingSet)
    #For accuracy counting
    correct = 0

    #Evaluate the model performance and print the predictions
    for doc, true_label in testSet:
      predicted_label = classifier.classify(doc)
      print(f"document: ({doc!r}, {true_label!r})\t predicted class: {predicted_label}")

      if predicted_label == true_label:
        #Add to accuracy if it's correct
        correct += 1

    #Compute the accuracy
    accuracy = correct / len(testSet)
    print("\nAccuracy:", accuracy)

trainingSet = [('London is the Capital of GB','GB'),
  ('Oxford is a city in GB','GB'),
  ('Dublin is the capital of Ireland','IE'),
  ('Limerick is a city in Ireland','IE')]

testSet = [('University of Limerick','IE'),
  ('University College Dublin','IE'),
  ('Imperial College London','GB'),
  ('University of Oxford','GB'),
  ('Ireland & GB','IE')]

naiveBayesClassifier(trainingSet,testSet)

"""Answer to C: The Naive Bayes classifier predicts the class based on the probability of each word occurring in each class. The last document contains the words: "Ireland" and "GB". TextBlob counts how often these words appear in the training data, and since the sentence contains one word from each class, the final predictions depends on a few things. These include Laplace smoothing, which word appears more often in the training data, tokenization behaviour, and the prior probability of each class. Essentially, the classifier is uncertain because the text contains words associated with both classes."""

