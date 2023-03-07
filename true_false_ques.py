# To run
# input: a string
# List[List[]] 
# index 0 of every inner List will be the true statement
# Syntax Ex: res_complete = true_false_generation(text)

import torch
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
import spacy
from string import punctuation
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
from nltk import tokenize
import scipy
torch.manual_seed(2020)

from spacy.cli import download
download("en_core_web_sm")

nlp = spacy.load("en_core_web_sm")
nltk.download('punkt')
benepar.download('benepar_en3')
benepar_parser = benepar.Parser("benepar_en3")

def preprocess(sentences):
    output = []
    for sent in sentences:
        single_quotes_present = len(re.findall(r"['][\w\s.:;,!?\\-]+[']",sent))>0
        double_quotes_present = len(re.findall(r'["][\w\s.:;,!?\\-]+["]',sent))>0
        question_present = "?" in sent
        if single_quotes_present or double_quotes_present or question_present :
            continue
        else:
            output.append(sent.strip(punctuation))
    return output

def get_candidate_sents(resolved_text, ratio=0.3):
    #candidate_sents = summarize(resolved_text, ratio=ratio)
    #candidate_sents_list = tokenize.sent_tokenize(candidate_sents)
    candidate_sents_list = tokenize.sent_tokenize(resolved_text)
    candidate_sents_list = [re.split(r'[:;]+',x)[0] for x in candidate_sents_list ]
    # Remove very short sentences less than 30 characters and long sentences greater than 150 characters
    filtered_list_short_sentences = [sent for sent in candidate_sents_list if len(sent)>30 and len(sent)<150]
    return filtered_list_short_sentences
    
def get_flattened(t):
    sent_str_final = None
    if t is not None:
        sent_str = [" ".join(x.leaves()) for x in list(t)]
        sent_str_final = [" ".join(sent_str)]
        sent_str_final = sent_str_final[0]
    return sent_str_final
    

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
    
def get_right_most_VP_or_NP(parse_tree,last_NP = None,last_VP = None):
    if len(parse_tree.leaves()) == 1:
        return get_flattened(last_NP),get_flattened(last_VP)
    last_subtree = parse_tree[-1]
    if last_subtree.label() == "NP":
        last_NP = last_subtree
    elif last_subtree.label() == "VP":
        last_VP = last_subtree
    
    return get_right_most_VP_or_NP(last_subtree,last_NP,last_VP)


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

    # Actiavte top_k sampling and top_p sampling with only from 90% most likely words
    sample_outputs = model.generate(
        input_ids, 
        do_sample=True, 
        max_length=maximum_length, 
        top_p=0.90, # 0.85 
        top_k=50,   #0.30
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
    
def get_flattened(t):
    sent_str_final = None
    if t is not None:
        sent_str = [" ".join(x.leaves()) for x in list(t)]
        sent_str_final = [" ".join(sent_str)]
        sent_str_final = sent_str_final[0]
    return sent_str_final
    

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
    
def get_right_most_VP_or_NP(parse_tree,last_NP = None,last_VP = None):
    if len(parse_tree.leaves()) == 1:
        return get_flattened(last_NP),get_flattened(last_VP)
    last_subtree = parse_tree[-1]
    if last_subtree.label() == "NP":
        last_NP = last_subtree
    elif last_subtree.label() == "VP":
        last_VP = last_subtree
    
    return get_right_most_VP_or_NP(last_subtree,last_NP,last_VP)


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
    
    # print(text)
    cand_sents = get_candidate_sents(text)
    # print(cand_sents)
    filter_quotes_and_questions = preprocess(cand_sents)
    # print(filter_quotes_and_questions)
    #for each_sentence in filter_quotes_and_questions:
    #    print (each_sentence)
    #    print ("\n")
    sent_completion_dict = get_sentence_completions(filter_quotes_and_questions)
    # print(sent_completion_dict)

    index = 1
    # choice_list = ["a)","b)","c)","d)","e)","f)"]
    res_complete = []
    for key_sentence in sent_completion_dict:
        res_individual = []
        partial_sentences = sent_completion_dict[key_sentence]
        false_sentences =[]
        #print_string = "**%s) True Sentence (from the story) :**"%(str(index))
        #print(print_string)
        #print ("  ",key_sentence)
        res_individual.append(str(key_sentence))
        for partial_sent in partial_sentences:
            false_sents = generate_sentences(partial_sent,key_sentence, model, model_BERT, tokenizer)
            false_sentences.extend(false_sents)
        res_individual.extend(false_sents)
        res_complete.append(res_individual)
    #    print("  **False Sentences (GPT-2 Generated)**")
    #    for ind,false_sent in enumerate(false_sentences):
    #        print_string_choices = "**%s** %s"%(choice_list[ind],false_sent)
    #        print(print_string_choices)
        index = index+1  
    #   print ("\n\n")
    return res_complete

if __name__ == "__main__":
    x = true_false_generation('''Stellantis revealed during CES 2023 its answer to an increasingly crowded battery-electric truck market: A broad-shouldered pickup loaded with tech, a longer cabin with third-row jump seats, cup holders in the frunk and even a movie projector.
The Ram 1500 Revolution BEV concept isn’t exactly what the Stellantis brand plans to put into production by 2024. (That version will be shown later this year). Still, it provides the clearest picture yet of Ram’s plans for its next-generation of trucks and how it aims to compete with other entrants in the nascent EV truck market, including the Ford F-150 Lightning and the Chevrolet Silverado EV.''')
    print(x)
