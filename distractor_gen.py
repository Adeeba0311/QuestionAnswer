#!pip install --quiet sense2vec


# load sense2vec vectors
from sense2vec import Sense2Vec
s2v = Sense2Vec().from_disk('models/s2v_reddit_2015_md/s2v_old')

from sentence_transformers import SentenceTransformer
model= SentenceTransformer('all-MiniLM-L12-v2')

def get_answer_and_distractor_embeddings(answer,candidate_distractors):
  answer_embedding = model.encode([answer])
  distractor_embeddings = model.encode(candidate_distractors)
  return answer_embedding,distractor_embeddings

from typing import List, Tuple
import itertools
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def mmr(doc_embedding: np.ndarray,
        word_embeddings: np.ndarray,
        words: List[str],
        top_n: int = 5,
        diversity: float = 0.9) -> List[Tuple[str, float]]:

    # Extract similarity within words, and between words and the document
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    # Initialize candidates and already choose best keyword/keyphras
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # Calculate MMR
        mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]
        # print(words[mmr_idx], mmr)

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [(words[idx], round(float(word_doc_similarity.reshape(1, -1)[0][idx]), 4)) for idx in keywords_idx]
    
  
def generate_distractors(originalword):
  word = originalword.lower()
  word = word.replace(" ", "_")

  #print ("word ",word)
  sense = s2v.get_best_sense(word)
  #print ("Best sense ", sense)
  most_similar = s2v.most_similar(sense, n=20)
  #print (most_similar)
  distractors = []

  for each_word in most_similar:
    append_word = each_word[0].split("|")[0].replace("_", " ")
    if append_word not in distractors and append_word != originalword:
        distractors.append(append_word)

  #print (distractors)
  distractors.insert(0,originalword)
  # print (distractors)
  answer_embedd, distractor_embedds = get_answer_and_distractor_embeddings(originalword,distractors)
  final_distractors = mmr(answer_embedd,distractor_embedds, distractors,5, 0.5)
  filtered_distractors = []
  for dist in final_distractors:
    filtered_distractors.append (dist[0])
  
  answer = filtered_distractors[0]
  filtered_Distractors =  filtered_distractors[1:]
  
  #print (Answer)
  #print ("------------------->")
  #for k in Filtered_Distractors:
  #  print (k)
  return filtered_Distractors

if __name__ == "__main__":
  generate_distractors("network layer")