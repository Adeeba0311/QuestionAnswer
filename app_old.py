from flask import Flask, render_template,request
import pandas as pd
import PyPDF2
from pprint import pprint
from Questgen import main
from transformers import T5Tokenizer, T5ForConditionalGeneration

# initialize the model architecture and weights
model = T5ForConditionalGeneration.from_pretrained("t5-large")
# initialize the model tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-large")

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def predict():
    json_result = []
    if request.method == 'POST':
        myfile = request.form['myfile']
        print("myfile *** " + myfile);
        print("****************ARTICLE*********************")
        # creating a pdf file object
        pdfFileObj = open(myfile, 'rb')

        # creating a pdf reader object
        pdfReader = PyPDF2.PdfReader(pdfFileObj)

        # printing number of pages in pdf file
        print(len(pdfReader.pages))

        article = ""
        # creating a page object
        for i in range(0,len(pdfReader.pages)):
            pageObj = pdfReader.pages[i]

            # extracting text from page 
            article = article + pageObj.extract_text()
            article.replace("\n","")

        # closing the pdf file object
        pdfFileObj.close()
        
        print("*****************summary_text********************")
        # payload = {"input_text": ARTICLE}
        # summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        # encode the text into tensor of integers using the tokenizer
        inputs = tokenizer.encode(article, return_tensors="pt", max_length=512, padding="max_length", truncation=True)

        summary_ids = model.generate(inputs,num_beams=int(2),no_repeat_ngram_size=3,length_penalty=2.0,min_length=100,max_length=200,early_stopping=True)

        outputSummary = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

        payload = {"input_text": article}
        qg = main.QGen()
        output = qg.predict_mcq(payload)
        colname = ['question_statement', 'MCQ']
        for  i in output['questions']:
            json_result.append({'question_statement': i['question_statement'], 'MCQ': i['options']})
        print (json_result)
    return render_template('result.html', prediction = json_result, colnames = colname, summary= outputSummary)

if __name__ == '__main__':
    app.run()
