

#Task 1 Etivity 3
#Use the following code snippet to download unigrams and bigrams data, and load them into their corresponding dataframes, unigrams_df and bigrams_df
from IPython.core.display import display, HTML
import pandas as pd
import math

!wget https://norvig.com/ngrams/count_1w.txt #unigram data
!wget https://norvig.com/ngrams/count_2w.txt #bigram data

filePath1 = "/content/count_1w.txt"
filePath2 = "/content/count_2w.txt"

unigrams_df = pd.read_csv(filePath1,sep='\t',header=None, names=['unigram','count'])
bigrams_df = pd.read_csv(filePath2,sep='\t',header=None, names=['bigram','count'])

print(unigrams_df.shape, bigrams_df.shape)
print(f'Number of unigrams: {unigrams_df.shape[0]} Number of Bigrams: {bigrams_df.shape[0]}')  # df.size VS df.shape

display(unigrams_df.head(100),bigrams_df.head(100))

# Convert columns to string type to handle potential non-string entries
unigrams_df['unigram'] = unigrams_df['unigram'].astype(str)
bigrams_df['bigram'] = bigrams_df['bigram'].astype(str)

#Build lookup dictionaries and make sure words are normalised
unigram_counts = dict(zip(unigrams_df['unigram'], unigrams_df['count']))
bigram_counts = dict(zip(bigrams_df['bigram'], bigrams_df['count']))


#Return the sentence using bigram chain rule (no add-one smoothing)
#Return 0 if the unigram or bigram is missing
def probability(sentence):
  words = sentence.lower().split()
  #If there are no words at all
  if len(words) == 0:
    return 0.0
  #If a single word, we can return the unigram probability
  if len(words) == 1:
    w = words[0]
    if unigram_counts.get(w, 0) > 0:
      prob = unigram_counts.get(w, 0) / sum(unigram_counts.values())
      print(f'Single word sentence: "{w}"')
      print(f'Unigram count for "{w}" = {unigram_counts[w]}')
      print(f'Total unigram count = {sum(unigram_counts.values())}')
      print(f'Unigram probability for "{w}" = {prob}\n')
      return prob
    else:
      return 0.0

  #Sentence info:
  print(f'Sentence: {words}\n')
  #Use log-prob to avoid overflow
  log_prob = 0.0

  #If there are two or more words
  for i in range(1, len(words)):
    w1 = words[i-1].strip()
    w2 = words[i].strip()
    bigram_key = f"{w1} {w2}"

    #Get the counts
    count_bigram = bigram_counts.get(bigram_key, 0)
    count_unigram = unigram_counts.get(w1, 0)
    print(f"{w1} {w2}")
    print(f' Bigram count for "{bigram_key}" = {count_bigram}')
    print(f' Unigram count for "{w1}" = {count_unigram}')

    #If the word is OOV OR bigram missing, prob is 0
    if count_unigram == 0:
      #word is not in vocabulary
      print(f' "{w1}" is OOV, no unigram is found, probability is 0\n')
      return 0.0
    if count_bigram == 0:
      print(f' Bigram "{bigram_key}" unseen, probability is 0\n')
      #Unseen bigram--no smoothing allowed here
      return 0.0

    #Get the probability for the bigram count
    bigram_prob = count_bigram / count_unigram
    print(f' Bigram probability for "{bigram_key}" = {bigram_prob}\n')

    #Accumulate log prob
    log_prob += math.log(count_bigram / count_unigram)

  #Convert back from log space
  total_prob = math.exp(log_prob)
  print(f'Total probability = {total_prob}\n')
  return total_prob

#TESTING
probA = probability("i love you")
brobB = probability('i hate you')
print(f'probability("i love you")>probability("i hate you") is: {probA>brobB}\n')
probability("OOV1 OOV2") # Test for OOV (Out Of Vocabulary) N-grams

#Task 1 cont. in new cell
from IPython.core.display import display, HTML
import pandas as pd
import math

!wget https://norvig.com/ngrams/count_1w.txt #unigram data
!wget https://norvig.com/ngrams/count_2w.txt #bigram data

filePath1 = "/content/count_1w.txt"
filePath2 = "/content/count_2w.txt"

unigrams_df = pd.read_csv(filePath1,sep='\t',header=None, names=['unigram','count'])
bigrams_df = pd.read_csv(filePath2,sep='\t',header=None, names=['bigram','count'])

print(unigrams_df.shape, bigrams_df.shape)
print(f'Number of unigrams: {unigrams_df.shape[0]} Number of Bigrams: {bigrams_df.shape[0]}')  # df.size VS df.shape

display(unigrams_df.head(100),bigrams_df.head(100))

# Convert columns to string type to handle potential non-string entries
unigrams_df['unigram'] = unigrams_df['unigram'].astype(str)
bigrams_df['bigram'] = bigrams_df['bigram'].astype(str)

#Build lookup dictionaries and make sure words are normalised
unigram_counts = dict(zip(unigrams_df['unigram'], unigrams_df['count']))
bigram_counts = dict(zip(bigrams_df['bigram'], bigrams_df['count']))

#Function for add-one smoothing
def probability_addone(sentence):
  #Sentence proabability, but with smoothing, unlike previous cell
  words = sentence.lower().split()
  #If there are no words at all
  if len(words) == 0:
    return 0.0

  #Number of unique unigrams
  unique_unigrams = len(unigram_counts)
  print(f"\nSentence = {words}\n")
  #Set log prob to 0
  log_prob = 0.0

  #Same info as before if two or more words
  for i in range(1, len(words)):
    w1 = words[i-1].strip()
    w2 = words[i].strip()
    bigram_key = f"{w1} {w2}"

    #Get counts
    count_bigram = bigram_counts.get(bigram_key, 0)
    count_unigram = unigram_counts.get(w1, 0)

    #Add the add-one smoothing formula
    smooth_prob = (count_bigram + 1) / (count_unigram + unique_unigrams)

    #Print statements for debug checks
    print(f"{w1} {w2}")
    print(f' Bigram count for "{bigram_key}" = {count_bigram}')
    print(f' Unigram count for "{w1}" = {count_unigram}')
    print(f' Smooth probability for "{bigram_key}" = {smooth_prob}\n')

    #Accumulate log probability
    log_prob += math.log(smooth_prob)

  total_prob = math.exp(log_prob)
  print(f'Total smooth probability = {total_prob}\n')
  return total_prob


#TESTING
probA = probability_addone("i love you")
brobB = probability_addone('i hate you')
print(f'probability_addone("i love you")>probability_addone("i hate you") is: {probA>brobB}\n')
probability_addone("OOV1 OOV2") # Test for OOV (Out Of Vocabulary) N-grams

#Task 2
import random
from IPython.core.display import display, HTML
import pandas as pd

!wget https://norvig.com/ngrams/count_1w.txt #unigram data
!wget https://norvig.com/ngrams/count_2w.txt #bigram data


filePath1 = "/content/count_1w.txt"
filePath2 = "/content/count_2w.txt"


unigrams_df = pd.read_csv(filePath1,sep='\t',header=None, names=['unigram','count'])
bigrams_df = pd.read_csv(filePath2,sep='\t',header=None, names=['bigram','count'])

#Added max words so it can't get too big
def ShannonVisualization(seed="<S>", max_words=20):

    current_word = seed.lower().strip()
    sentence = [current_word]

    for _ in range(max_words - 1):  # generate up to max_words
        #Find all bigrams that start with current_word
        candidates = bigrams_df[bigrams_df['bigram'].str.startswith(current_word + " ")].copy()

        #If no candidates, stop
        if candidates.empty:
            break

        #Extract next words and their counts
        candidates['next_word'] = candidates['bigram'].apply(lambda b: b.split()[1])
        next_words = candidates['next_word'].tolist()
        counts = candidates['count'].tolist()

        #Compute cumulative prob intervals
        total_count = sum(counts)
        probs = [c / total_count for c in counts]
        cumulative = []
        cumulative_sum = 0.0
        for p in probs:
            # Fix: Append a tuple instead of two arguments
            cumulative.append((cumulative_sum, cumulative_sum + p))
            cumulative_sum += p

        #Pick a random number and find which interval it falls into
        r = random.random()
        chosen_word = None  # Initialize chosen_word
        for i, (low, high) in enumerate(cumulative):
            if low <= r < high:
                chosen_word = next_words[i]
                break

        # If no word was chosen (shouldn't happen with correct probabilities, but as a safeguard)
        if chosen_word is None:
            break

        #Append and move on
        sentence.append(chosen_word)
        current_word = chosen_word

    print("Generated sentence:")
    print(" ".join(sentence))

ShannonVisualization("i")