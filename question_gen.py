#!pip install sentence-transformers transformers

from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer, BertTokenizer, BertModel, AutoTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import spacy
from transformers import BertTokenizer, BertModel
from warnings import filterwarnings as filt

filt('ignore')

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained("bert-base-uncased")
model = SentenceTransformer('distilbert-base-nli-mean-tokens')
nlp = spacy.load("en_core_web_sm")

def get_question(sentence, answer):

  mdl = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
  tknizer = AutoTokenizer.from_pretrained('ramsrigouthamg/t5_squad_v1')

  text = "context: {} answer: {}".format(sentence,answer)
  max_len = 256
  encoding = tknizer.encode_plus(text,max_length=max_len, pad_to_max_length=False,truncation=True, return_tensors="pt")

  input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

  outs = mdl.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  early_stopping=True,
                                  num_beams=5,
                                  num_return_sequences=1,
                                  no_repeat_ngram_size=2,
                                  max_length=300)


  dec = [tknizer.decode(ids,skip_special_tokens=True) for ids in outs]


  Question = dec[0].replace("question:","")
  Question= Question.strip()
  return Question
  
def get_embedding(doc):

  bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  bert_model = BertModel.from_pretrained("bert-base-uncased")
  
  # txt = '[CLS] ' + doc + ' [SEP]'
  tokens = bert_tokenizer.tokenize(txt)
  token_idx = bert_tokenizer.convert_tokens_to_ids(tokens)
  segment_ids = [1] * len(tokens)

  torch_token = torch.tensor([token_idx])
  torch_segment = torch.tensor([segment_ids])

  return bert_model(torch_token, torch_segment)[-1].detach().numpy()

def get_pos(context):
  doc = nlp(context)
  docs = [d.pos_ for d in doc]
  return docs, context.split()

def get_sent(context):
  doc = nlp(context)
  return list(doc.sents)

def get_vector(doc):
  stop_words = "english"
  n_gram_range = (1,1)
  df = CountVectorizer(ngram_range = n_gram_range, stop_words = stop_words).fit([doc])
  return df.get_feature_names()


def get_key_words(context, module_type = 't'):
  keywords = []
  top_n = 5
  for txt in get_sent(context):
    keywd = get_vector(str(txt))
    print(f'vectors : {keywd}')
    if module_type == 't':
      doc_embedding = get_embedding(str(txt))
      keywd_embedding = get_embedding(' '.join(keywd))
    else:
      doc_embedding = model.encode([str(txt)])
      keywd_embedding = model.encode(keywd)
    
    distances = cosine_similarity(doc_embedding, keywd_embedding)
    print(distances)
    keywords += [(keywd[index], str(txt)) for index in distances.argsort()[0][-top_n:]]

  return keywords
 
if __name__ == "__main__":
  # test
  txt = 'Tim Berners-Lee, a British scientist, invented the World Wide Web (WWW) in 1989, while working at CERN. The web was originally conceived and developed to meet the demand for automated information-sharing between scientists in universities and institutes around the world.'
  for ans, context in get_key_words(txt, 'st'):
    print('=======================================')
    print()
    print(get_question(context, ans))
    print()