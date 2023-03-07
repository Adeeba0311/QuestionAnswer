#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#pip install PyPDF2==3.0.1
#pip install transformers==4.25.1


# In[25]:


import PyPDF2
from tkinter.filedialog import askopenfilename
   
# creating a pdf file object
pdfFileObj = open(askopenfilename(filetypes=[("*","*.pdf")]), 'rb')
   
# creating a pdf reader object
pdfReader = PyPDF2.PdfReader(pdfFileObj)
   
# printing number of pages in pdf file
ARTICLE = []
str=""
print(len(pdfReader.pages))
for i in range(len(pdfReader.pages)): 
    # creating a page object
    pageObj = pdfReader.pages[i]

    # extracting text from page
    ARTICLE.append(pageObj.extract_text().replace('\n', ' '))
    #str+=pageObj.extract_text()
    
#print(ARTICLE)
print(len(ARTICLE))

   
# closing the pdf file object
pdfFileObj.close()


# In[ ]:





# In[26]:


from transformers import T5Tokenizer, T5ForConditionalGeneration

# initialize the model architecture and weights
model = T5ForConditionalGeneration.from_pretrained("t5-base")
# initialize the model tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-base")

summ = ""
for article in ARTICLE:
    print("**************")
    # encode the text into tensor of integers using the tokenizer
    inputs = tokenizer.encode(article, return_tensors="pt", max_length=512, padding="max_length", truncation=True)
    summary_ids = model.generate(inputs,num_beams=int(2),no_repeat_ngram_size=3,length_penalty=2.0,min_length=100,max_length=200,early_stopping=True)
    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    # res = summarizer(article, max_length=130, min_length=30, do_sample=False)
    print(output)
    summ = summ + output


# In[22]:





# In[ ]:




