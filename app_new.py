
import os
# os.environ['TRANSFORMERS_CACHE'] = 'D:\\temp\\models']

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from flask import Flask, render_template,request, redirect
import PyPDF2
import speech_recognition as sr
#from tkinter.filedialog import askopenfilename
#from pprint import pprint
#from Questgen import main
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer, BertTokenizer, BertModel, AutoTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import spacy
from transformers import BertTokenizer, BertModel
from warnings import filterwarnings as filt

from typing import List, Tuple
import itertools
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import transformers
import requests
import json
from summa.summarizer import summarize
import benepar
import string
import nltk
from nltk import tokenize
from nltk.tokenize import sent_tokenize
import re
from random import shuffle
from string import punctuation
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
import scipy
torch.manual_seed(2020)
import en_core_web_sm
from spacy.cli import download


# download("en_core_web_sm")
from sense2vec import Sense2Vec
s2v = Sense2Vec().from_disk('../s2v_old')

nlp = en_core_web_sm.load()
nltk.download('punkt')
benepar.download('benepar_en3')
benepar_parser = benepar.Parser("benepar_en3")

# initialize the model architecture and weights
modelT5 = T5ForConditionalGeneration.from_pretrained("t5-base")
# initialize the model tokenizer
tokenizerT5 = T5Tokenizer.from_pretrained("t5-base")

filt('ignore')

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained("bert-base-uncased")
modelDB = SentenceTransformer('distilbert-base-nli-mean-tokens')

# nlp = spacy.load("en_core_web_sm")

from sentence_transformers import SentenceTransformer
modelL12= SentenceTransformer('all-MiniLM-L12-v2')

app = Flask(__name__)
@app.route('/')
def home():
    #return render_template('home.html')
    return redirect('/login')

@app.route('/predict', methods = ['POST'])
def predict():
    json_result = []
    if request.method == 'POST':
        myfile = request.form['myfile']
        
        articles = get_text_pdf(myfile)

        summ = ""
        for article in articles:
            print("**************")
            # encode the text into tensor of integers using the tokenizer
            inputs = tokenizerT5.encode(article, return_tensors="pt", max_length=512, padding="max_length", truncation=True)
            summary_ids = modelT5.generate(inputs,num_beams=int(2),no_repeat_ngram_size=3,length_penalty=2.0,min_length=100,max_length=200,early_stopping=True)
            output = tokenizerT5.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            print(output)
            summ = summ + output

        qas = generate_ques(summ)

        mcq = []

        for ques_ans in qas:
            distractors = generate_distractors(ques_ans[1])
            mcq.append((ques_ans[0], ques_ans[1], distractors))

        colname = ['Question', 'Answer', 'Distractors']
        return render_template('result.html', prediction=mcq, colnames=colname, summary=summ)        
            

    
        #payload = {"input_text": article}
        #qg = main.QGen()
        #output = qg.predict_mcq(payload)
        #colname = ['question_statement', 'MCQ']
        #for  i in output['questions']:
        #    json_result.append({'question_statement': i['question_statement'], 'MCQ': i['options']})
        #print (json_result)
    #return render_template('result.html', prediction = json_result, colnames = colname, summary= outputSummary)

### *******************Question/Generation**************######
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

    tokens = bert_tokenizer.tokenize(doc)
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
            doc_embedding = modelDB.encode([str(txt)])
            keywd_embedding = modelDB.encode(keywd)
        
        distances = cosine_similarity(doc_embedding, keywd_embedding)
        print(distances)
        keywords += [(keywd[index], str(txt)) for index in distances.argsort()[0][-top_n:]]

    return keywords

def generate_ques(text):
    qs = []
    for ans, context in get_key_words(text, 'st'):
        qs.append((get_question(context, ans), ans))
    return qs



####********Multiple Choice Generation*******#######
def get_answer_and_distractor_embeddings(answer,candidate_distractors):
    answer_embedding = modelL12.encode([answer])
    distractor_embeddings = modelL12.encode(candidate_distractors)
    return answer_embedding,distractor_embeddings

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
    if sense is None:
        return []
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
    final_distractors = mmr(answer_embedd,distractor_embedds, distractors,5, 0.6)
    filtered_distractors = []
    for dist in final_distractors:
        filtered_distractors.append (dist[0])

    answer = filtered_distractors[0]
    filtered_Distractors =  filtered_distractors[1:]

    return filtered_Distractors

#*********Boolean Ques/Ans********#

def preprocess_cand_sents(sentences):
    output = []
    for sent in sentences:
        single_quotes = len(re.findall(r"['][\w\s.:;,!?\\-]+[']",sent))>0
        double_quotes = len(re.findall(r'["][\w\s.:;,!?\\-]+["]',sent))>0
        question_present = "?" in sent
        if single_quotes or double_quotes or question_present :
            continue
        else:
            output.append(sent.strip(punctuation))
    return output

def generate_candidate_sents(text):
    cand_sents = tokenize.sent_tokenize(text)
    cand_sents = [re.split(r'[:;]+',x)[0] for x in cand_sents ]
    short_sentences = [sent for sent in cand_sents if len(sent)>30 and len(sent)<150]
    return short_sentences
    
def flatten_sents(t):
    sent_final = None
    if t is not None:
        sent_str = [" ".join(x.leaves()) for x in list(t)]
        sent_final = [" ".join(sent_str)]
        sent_final = sent_final[0]
    return sent_final
    

def get_termination_portion(main_string,sub_string):
    combined_sub_string = sub_string.replace(" ","")
    main_string_list = main_string.split()
    last_index = len(main_string_list)
    for i in range(last_index):
        check_string_list = main_string_list[i:]
        check_string = "".join(check_string_list)
        check_string = check_string.replace(" ","")
        if check_string == combined_sub_string:
            return " ".join(main_string_list[:i])
                     
    return None
    
####
## ramsrigouthamg, Generate_True_or_False_OpenAI_GPT2_Sentence_BERT, (2020), GitHub repository, https://github.com/ramsrigouthamg/Generate_True_or_False_OpenAI_GPT2_Sentence_BERT
####
def get_right_most_VP_or_NP(parse_tree,last_NP = None,last_VP = None):
    if len(parse_tree.leaves()) == 1:
        return flatten_sents(last_NP),flatten_sents(last_VP)
    last_subtree = parse_tree[-1]
    if last_subtree.label() == "NP":
        last_NP = last_subtree
    elif last_subtree.label() == "VP":
        last_VP = last_subtree
    
    return get_right_most_VP_or_NP(last_subtree,last_NP,last_VP)


####
## ramsrigouthamg, Generate_True_or_False_OpenAI_GPT2_Sentence_BERT, (2020), GitHub repository, https://github.com/ramsrigouthamg/Generate_True_or_False_OpenAI_GPT2_Sentence_BERT
####
def get_sentence_completions(filter_quotes_and_questions):
    sentence_completion_dict = {}
    for individual_sentence in filter_quotes_and_questions:
        sentence = individual_sentence.rstrip('?:!.,;')
        tree = benepar_parser.parse(sentence)
        last_nounphrase, last_verbphrase =  get_right_most_VP_or_NP(tree)
        phrases= []
        if last_verbphrase is not None:
            verbphrase_string = get_termination_portion(sentence,last_verbphrase)
            phrases.append(verbphrase_string)
        if last_nounphrase is not None:
            nounphrase_string = get_termination_portion(sentence,last_nounphrase)
            phrases.append(nounphrase_string)

        longest_phrase =  sorted(phrases, key=len,reverse= True)
        if len(longest_phrase) == 2:
            first_sent_len = len(longest_phrase[0].split())
            second_sentence_len = len(longest_phrase[1].split())
            if (first_sent_len - second_sentence_len) > 4:
                del longest_phrase[1]
                
        if len(longest_phrase)>0:
            sentence_completion_dict[sentence]=longest_phrase
    return sentence_completion_dict

####
## ramsrigouthamg, Generate_True_or_False_OpenAI_GPT2_Sentence_BERT, (2020), GitHub repository, https://github.com/ramsrigouthamg/Generate_True_or_False_OpenAI_GPT2_Sentence_BERT
####
def sort_by_similarity(original_sentence,generated_sentences_list, model_BERT):
    # Each sentence is encoded as a 1-D vector with 768 columns
    sentence_embeddings = model_BERT.encode(generated_sentences_list)

    queries = [original_sentence]
    query_embeddings = model_BERT.encode(queries)
    # Find the top sentences of the corpus for each query sentence based on cosine similarity
    number_top_matches = len(generated_sentences_list)

    dissimilar_sentences = []

    for query, query_embedding in zip(queries, query_embeddings):
        distances = scipy.spatial.distance.cdist([query_embedding], sentence_embeddings, "cosine")[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])


        for idx, distance in reversed(results[0:number_top_matches]):
            score = 1-distance
            if score < 0.9:
                dissimilar_sentences.append(generated_sentences_list[idx].strip())
           
    sorted_dissimilar_sentences = sorted(dissimilar_sentences, key=len)
    
    return sorted_dissimilar_sentences[:3]
    

def generate_sentences(partial_sentence,full_sentence, model, model_BERT, tokenizer):
    input_ids = torch.tensor([tokenizer.encode(partial_sentence)])
    maximum_length = len(partial_sentence.split())+80

    sample_outputs = model.generate(
        input_ids, 
        do_sample=True, 
        max_length=maximum_length, 
        top_p=0.90,
        top_k=50,
        repetition_penalty  = 10.0,
        num_return_sequences=10
    )
    generated_sentences=[]
    for i, sample_output in enumerate(sample_outputs):
        decoded_sentences = tokenizer.decode(sample_output, skip_special_tokens=True)
        decoded_sentences_list = tokenize.sent_tokenize(decoded_sentences)
        generated_sentences.append(decoded_sentences_list[0])
        
    top_3_sentences = sort_by_similarity(full_sentence,generated_sentences, model_BERT)
    
    return top_3_sentences  

def get_termination_portion(main_string,sub_string):
    combined_sub_string = sub_string.replace(" ","")
    main_string_list = main_string.split()
    last_index = len(main_string_list)
    for i in range(last_index):
        check_string_list = main_string_list[i:]
        check_string = "".join(check_string_list)
        check_string = check_string.replace(" ","")
        if check_string == combined_sub_string:
            return " ".join(main_string_list[:i])
                     
    return None

def get_sentence_completions(filter_quotes_and_questions):
    sentence_completion_dict = {}
    for individual_sentence in filter_quotes_and_questions:
        sentence = individual_sentence.rstrip('?:!.,;')
        tree = benepar_parser.parse(sentence)
        last_nounphrase, last_verbphrase =  get_right_most_VP_or_NP(tree)
        phrases= []
        if last_verbphrase is not None:
            verbphrase_string = get_termination_portion(sentence,last_verbphrase)
            phrases.append(verbphrase_string)
        if last_nounphrase is not None:
            nounphrase_string = get_termination_portion(sentence,last_nounphrase)
            phrases.append(nounphrase_string)

        longest_phrase =  sorted(phrases, key=len,reverse= True)
        if len(longest_phrase) == 2:
            first_sent_len = len(longest_phrase[0].split())
            second_sentence_len = len(longest_phrase[1].split())
            if (first_sent_len - second_sentence_len) > 4:
                del longest_phrase[1]
                
        if len(longest_phrase)>0:
            sentence_completion_dict[sentence]=longest_phrase
    return sentence_completion_dict
  
################################################
# Driver Function 
################################################
def true_false_generation(text):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model_BERT = SentenceTransformer('bert-base-nli-mean-tokens')
    model = GPT2LMHeadModel.from_pretrained("gpt2",pad_token_id=tokenizer.eos_token_id)
    text = text.replace('\n','')
    text = text.replace('\t',' ')

    cand_sents = generate_candidate_sents(text)
    cand_sents = preprocess_cand_sents(cand_sents)
    sentence_completion_dict = get_sentence_completions(cand_sents)

    res_complete = []
    for key_sentence in sentence_completion_dict:
        res_individual = []
        partial_sentences = sentence_completion_dict[key_sentence]
        false_sentences =[]

        res_individual.append(str(key_sentence))
        for partial_sent in partial_sentences:
            false_sents = generate_sentences(partial_sent,key_sentence, model, model_BERT, tokenizer)
            false_sentences.extend(false_sents)
        res_individual.extend(false_sents)
        res_complete.append(res_individual) 

    return res_complete


############### GET QUESTIONS ###############

def get_questions(textList):
    summ = ""
    for text in textList:
        print("**************")
        # encode the text into tensor of integers using the tokenizer
        t5_prepared_Text = "summarize: "+ text
        inputs = tokenizerT5.encode(t5_prepared_Text, return_tensors="pt", max_length=512, padding="max_length", truncation=True)
        summary_ids = modelT5.generate(inputs,num_beams=int(4),no_repeat_ngram_size=2,length_penalty=2.0,min_length=100,max_length=200,early_stopping=True)
        output = tokenizerT5.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print(output)
        summ = summ + output

    qas = generate_ques(summ)
    mcq = []
    for ques_ans in qas:
        distractors = generate_distractors(ques_ans[1])
        distractors = distractors[0:3:1]
        distractors.append(ques_ans[1])
        shuffle(distractors)
        mcq.append((ques_ans[0], ques_ans[1], distractors))

    return mcq

def get_questions_video(videoFilepath):
    text = []
    temp = os.path.splitext(videoFilepath)
    command = "ffmpeg -i "+temp[0]+".mp4 "+temp[0]+".mp3"
    os.system(command)

    commandwav = "ffmpeg -i "+temp[0]+".mp3 "+temp[0]+".wav"
    os.system(commandwav)

    AUDIO_FILE = temp[0]+".wav"
    r = sr.Recognizer()
    audioFile = sr.AudioFile(AUDIO_FILE)

    with audioFile as source:
       audio = r.record(source, duration=100)

    print(type(audio))
    print("------------------------------------")
    text.append(r.recognize_google(audio))
    #text = r.recognize_sphinx(audio)
    #print(text)
    return get_questions(text)

def get_questions_audio(audioFilepath):
    text = []
    temp = os.path.splitext(audioFilepath)
   
    commandwav = "ffmpeg -i "+temp[0]+".mp3 "+temp[0]+".wav"
    os.system(commandwav)

    AUDIO_FILE = temp[0]+".wav"
    r = sr.Recognizer()
    audioFile = sr.AudioFile(AUDIO_FILE)

    with audioFile as source:
        audio = r.record(source, duration=100)

    print(type(audio))
    print("------------------------------------")
    text.append(r.recognize_google(audio))
    #text = r.recognize_sphinx(audio)
    #print(text)
    return get_questions(text)

def get_text_pdf(myfile):
    pdfFileObj = open(myfile, 'rb')
    
    pdfReader = PyPDF2.PdfReader(pdfFileObj)
    articles = []
    for i in range(len(pdfReader.pages)): 
        pageObj = pdfReader.pages[i]
        articles.append(pageObj.extract_text().replace('\n', ' '))
        print(len(articles))
    pdfFileObj.close()
    return articles

def get_questions_pdf(pdfFilepath):
    return get_questions(get_text_pdf(pdfFilepath))

from werkzeug.utils import secure_filename
import json

@app.route('/mcq', methods = ['GET','POST'])
def mcq_gen():
    filename = None
    mcq = None
    #if 1==1:
    #    mcq = json.loads("[[\"What type of systems does this paper compare the web with?\", \"contemporary\", [\"modern\", \"modernist\", \"contemporary\", \"contemporaries\"]], [\"What type of model does this paper describe?\", \"data\", [\"only data\", \"more data\", \"data\", \"relevant data\"]], [\"What type of data does this paper describe?\", \"model\", [\"most models\", \"actual model\", \"models\", \"model\"]], [\"Along with the aims and data model, what is needed to implement the web?\", \"protocols\", [\"network protocols\", \"protocols\", \"routing protocols\", \"new protocols\"]], [\"What does this paper describe the aims, data model, and protocols needed to implement?\", \"web\", [\"web services\", \"web\", \"websites\", \"Web\"]]]")
    #    return render_template("mcq.html", name = "abc.pdf", mcq = mcq)
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        filepath = os.path.join('tempFiles', filename)
        print("filepath::", filepath)
        file.save(filepath)
        filepath = os.path.abspath(filepath) # absolute
        filetype = request.form.get('filetype')
        print("filetype -- ", filetype)
        mcq = []
        if filetype == "pdf":
            mcq = get_questions_pdf(filepath)
        elif filetype == "video":
            mcq = get_questions_video(filepath)
        elif filetype == "audio":
            mcq = get_questions_audio(filepath)
        print(mcq)
    
    return render_template("mcq.html", name = filename, mcq = mcq)

@app.route('/login', methods = ['GET','POST'])
def login():
    return render_template("/login.html")

if __name__ == '__main__':
    app.run()
