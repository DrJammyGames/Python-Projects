#Task 1
import math

def multinomialNaiveBayesClassifier(trainingSet, testSet):
    #Count classes
    classCounts = {}
    for text, label in trainingSet:
      classCounts[label] = classCounts.get(label, 0) + 1

    print("posCount=", classCounts.get('+', 0), " negCount=", classCounts.get('-', 0))
    print()

    #Build a bag of words per class
    bow = {label: {} for label in classCounts}
    vocab = set()

    for text, label in trainingSet:
      for w in text.lower().split():
        vocab.add(w)
        bow[label][w] = bow[label].get(w, 0) + 1

    print("pos_BOW=", bow.get('+', {}))
    print("neg_BOW=", bow.get('-', {}))
    print()

    #Vocab info
    vocab = sorted(list(vocab))
    print("V =", vocab)
    print("|V| =", len(vocab))
    print()

    #Compute the priors
    total_docs = len(trainingSet)
    priors = {c: classCounts[c] / total_docs for c in classCounts}
    #Print the priors info
    print("probPos=", priors.get('+', 0), "probNeg=", priors.get('-', 0))
    print()

    #Conditional probabilities
    V = len(vocab)
    totalWords = {c: sum(bow[c].values()) for c in bow}
    condProb = {c: {} for c in bow}

    for c in bow:
      #Total docs in the class
      totalDocsWithWord = classCounts[c]
      for w in vocab:
        count = bow[c].get(w, 0)
        condProb[c][w] = (count + 1) / (totalWords[c] + V)

    #Classify the test documents and print nicely
    for text, _ in testSet:
      print("----------------------------------------------------")
      #Expected label unknown
      print(f"Test document=('{text}', '-')")
      print()

      #Word-by-word conditional probabilities
      for w in vocab:
        if w not in vocab:
          # Unseen word, use Laplace-smoothed uniform probability from before
          prob_pos = 1 / (totalWords['+'] + V)
          prob_neg = 1 / (totalWords['-'] + V)
        else:
          prob_pos = condProb['+'][w]
          prob_neg = condProb['-'][w]
        print(f"word= {w},\t wordConditionalProbPos= {prob_pos},\t wordConditionalProbNeg= {prob_neg}")
      print()

      #Log document probabilities
      docProbPos = math.log(priors['+'])
      docProbNeg = math.log(priors['-'])

      for w in vocab:
        if w not in vocab:
          docProbPos += math.log(1 / (totalWords['+'] + V))
          docProbNeg += math.log(1 / (totalWords['-'] + V))
        else:
          docProbPos += math.log(condProb['+'][w])
          docProbNeg += math.log(condProb['-'][w])

      print(f"docProbPos= {docProbPos:.6f},\t docProbNeg= {docProbNeg:.6f}")
      inferred = '+' if docProbPos > docProbNeg else '-'
      print("Inferred class=", inferred)


trainingSet = [
    ('Boxing scene was a disappointment','-'),
    ('No plot twists or great scenes','-'),
    ('Great satire and great plot twists','+'),
    ('Great scenes a great film','+')
]

testSet = [('Great disappointment indeed','?')]

#Test print
multinomialNaiveBayesClassifier(trainingSet,testSet)

#Task 1
import math

def multinomialNaiveBayesClassifier(trainingSet, testSet):
    #Count classes
    classCounts = {}
    for text, label in trainingSet:
        classCounts[label] = classCounts.get(label, 0) + 1

    print("posCount=", classCounts.get('+', 0), " negCount=", classCounts.get('-', 0))
    print()

    #Build a binary bag of words per class
    bow = {label: {} for label in classCounts}
    vocab = set()

    for text, label in trainingSet:
      #Only the presence of the word counts here
      words_in_text = set(text.lower().split())
      for w in words_in_text:
          vocab.add(w)
          bow[label][w] = bow[label].get(w, 0) + 1

    print("pos_BOW=", bow.get('+', {}))
    print("neg_BOW=", bow.get('-', {}))
    print()

    vocab = sorted(list(vocab))
    print("V =", vocab)
    print("|V| =", len(vocab))
    print()

    #Compute the priors
    total_docs = len(trainingSet)
    priors = {c: classCounts[c] / total_docs for c in classCounts}

    print("probPos=", priors.get('+', 0), "probNeg=", priors.get('-', 0))
    print()

    #Conditional probabilities
    V = len(vocab)
    totalWords = {c: sum(bow[c].values()) for c in bow}

    condProb = {c: {} for c in bow}

    for c in bow:
      #Total docs in the class
      totalDocsWithWord = classCounts[c]
      for w in vocab:
        count = bow[c].get(w, 0)
        #Binary Multinominal Naive Bayes algorithm instead of MNB
        #Add 2 because of binary
        condProb[c][w] = (count + 1) / (totalDocsWithWord + 2)

    #Classify the test documents and print nicely
    for text, _ in testSet:
      print("----------------------------------------------------")
      print(f"Test document=('{text}', '-')")
      print()

      #Only presence of a word matters
      words = set(text.lower().split())

      #Word-by-word conditional probabilities
      for w in vocab:
        if w in words:
          prob_pos = condProb['+'][w]
          prob_neg = condProb['-'][w]
        else:
          #Probability of absence: 1 - prob(word present)
          prob_pos = 1 - condProb['+'][w]
          prob_neg = 1 - condProb['-'][w]
        print(f"word= {w},\t wordConditionalProbPos= {prob_pos},\t wordConditionalProbNeg= {prob_neg}")
      print()

      #Log document probabilities
      docProbPos = math.log(priors['+'])
      docProbNeg = math.log(priors['-'])

      for w in vocab:
        if w in words:
          docProbPos += math.log(condProb['+'][w])
          docProbNeg += math.log(condProb['-'][w])
        else:
          docProbPos += math.log(1 - condProb['+'][w])
          docProbNeg += math.log(1 - condProb['-'][w])

      print(f"docProbPos= {docProbPos:.6f},\t docProbNeg= {docProbNeg:.6f}")
      inferred = '+' if docProbPos > docProbNeg else '-'
      print("Inferred class=", inferred)


trainingSet = [
    ('Boxing scene was a disappointment','-'),
    ('No plot twists or great scenes','-'),
    ('Great satire and great plot twists','+'),
    ('Great scenes a great film','+')
]

testSet = [('Great disappointment indeed','?')]

#Test print
multinomialNaiveBayesClassifier(trainingSet,testSet)

#Task 3 with performance metrics
import math

def binaryMultinomialNaiveBayesClassifier(trainingSet, testSet):
    #Count classes
    classCounts = {}
    for text, label in trainingSet:
        classCounts[label] = classCounts.get(label, 0) + 1

    print("posCount=", classCounts.get('+', 0), " negCount=", classCounts.get('-', 0))
    print()

    #Build a binary bag of words per class
    bow = {label: {} for label in classCounts}
    vocab = set()

    for text, label in trainingSet:
      #Only the presence of the word counts here
      words_in_text = set(text.lower().split())
      for w in words_in_text:
        vocab.add(w)
        bow[label][w] = bow[label].get(w, 0) + 1

    print("pos_BOW=", bow.get('+', {}))
    print("neg_BOW=", bow.get('-', {}))
    print()

    vocab = sorted(list(vocab))
    print("V =", vocab)
    print("|V| =", len(vocab))
    print()

    #Compute the priors
    total_docs = len(trainingSet)
    priors = {c: classCounts[c] / total_docs for c in classCounts}

    print("probPos=", priors.get('+', 0), "probNeg=", priors.get('-', 0))
    print()

    #Conditional probabilities
    V = len(vocab)
    totalWords = {c: sum(bow[c].values()) for c in bow}

    condProb = {c: {} for c in bow}

    for c in bow:
      #Total docs in the class
      totalDocsWithWord = classCounts[c]
      for w in vocab:
        count = bow[c].get(w, 0)
        #Binary Multinominal Naive Bayes algorithm instead of MNB
        #Add 2 because of binary
        condProb[c][w] = (count + 1) / (totalDocsWithWord + 2)

    #Add variables that start at 0 for computing
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    #Classify the test documents and print nicely
    #Now including the performance metrics
    for text, true_label in testSet:
      print("----------------------------------------------------")
      print(f"Test document=('{text}', '{true_label}')")
      print()

      #Only presence of a word matters
      words = set(text.lower().split())

      #Word-by-word conditional probabilities
      for w in vocab:
        if w in words:
          prob_pos = condProb['+'][w]
          prob_neg = condProb['-'][w]
        else:
          #Probability of absence: 1 - prob(word present)
          prob_pos = 1 - condProb['+'][w]
          prob_neg = 1 - condProb['-'][w]
        print(f"word= {w},\t wordConditionalProbPos= {prob_pos},\t wordConditionalProbNeg= {prob_neg}")
      print()

      #Log document probabilities
      docProbPos = math.log(priors['+'])
      docProbNeg = math.log(priors['-'])

      for w in vocab:
        if w in words:
          docProbPos += math.log(condProb['+'][w])
          docProbNeg += math.log(condProb['-'][w])
        else:
          docProbPos += math.log(1 - condProb['+'][w])
          docProbNeg += math.log(1 - condProb['-'][w])

      print(f"docProbPos= {docProbPos:.6f},\t docProbNeg= {docProbNeg:.6f}")
      inferred = '+' if docProbPos > docProbNeg else '-'
      print("Inferred class=", inferred)

      #Update performance metrics
      if inferred == '+' and true_label == '+':
        TP += 1
      elif inferred == '-' and true_label == '-':
        TN += 1
      elif inferred == '+' and true_label == '-':
        FP += 1
      elif inferred == '-' and true_label == '+':
        FN += 1

    print("----------------------------------------------------")
    print(f"TP= {TP}\t TN= {TN}")
    print(f"FP= {FP}\t FN= {FN}")
    print()

    #Performance metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

    #Print the performance metrics
    print(f"Accuracy= {accuracy:.6f}")
    print(f"Precision= {precision:.6f}")
    print(f"Recall= {recall:.6f}")
    print(f"F1= {f1:.6f}")

#Training and test sets
trainingSet = [
  ('Boxing scene was a disappointment','-'),
  ('No plot twists or great scenes','-'),
  ('Great satire and great plot twists','+'),
  ('Great scenes a great film','+')
]

testSet = [
  ('Great disappointment indeed','-'),
  ('What a great movie','+'),
  ('Great satire','+'),
  ('No plot twists or satire','-'),
  ('This movie was a disappointment','-'),
  ('great disappointment','-'),
  ('great boxing scenes','+'),
  ('great movie','+'),
  ('bad film','-'),
  ('nice film','+')
]

binaryMultinomialNaiveBayesClassifier(trainingSet, testSet)