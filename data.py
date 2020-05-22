
import pandas as pd
import numpy as np
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter
Height=500
Width=600
root = tkinter.Tk()


df1 = pd.read_csv('S08_question_answer_pairs.txt', sep='\t')
df2 = pd.read_csv('S09_question_answer_pairs.txt', sep='\t')
df3 = pd.read_csv('S10_question_answer_pairs.txt', sep='\t', encoding = 'ISO-8859-1')





df1.head(20)





all_data = df1.append([df2, df3])
all_data.info()





all_data['Question'] = all_data['ArticleTitle'].str.replace('_', ' ') + ' ' + all_data['Question']
all_data = all_data[['Question', 'Answer']]
all_data.shape





all_data.head(10)





all_data = all_data.drop_duplicates(subset='Question')
all_data.head(10)




all_data.shape




all_data = all_data.dropna()
all_data.shape




stopwords_list = stopwords.words('english')

lemmatizer = WordNetLemmatizer()

def my_tokenizer(doc):
    words = word_tokenize(doc)
    
    pos_tags = pos_tag(words)
    
    non_stopwords = [w for w in pos_tags if not w[0].lower() in stopwords_list]
    
    non_punctuation = [w for w in non_stopwords if not w[0] in string.punctuation]
    
    lemmas = []
    for w in non_punctuation:
        if w[1].startswith('J'):
            pos = wordnet.ADJ
        elif w[1].startswith('V'):
            pos = wordnet.VERB
        elif w[1].startswith('N'):
            pos = wordnet.NOUN
        elif w[1].startswith('R'):
            pos = wordnet.ADV
        else:
            pos = wordnet.NOUN
        
        lemmas.append(lemmatizer.lemmatize(w[0], pos))

    return lemmas





tfidf_vectorizer = TfidfVectorizer(tokenizer=my_tokenizer)
tfidf_matrix = tfidf_vectorizer.fit_transform(tuple(all_data['Question']))
print(tfidf_matrix.shape)
def ask_question(question):
    question=[question]
    query_vect = tfidf_vectorizer.transform(question)
    similarity = cosine_similarity(query_vect, tfidf_matrix)
    max_similarity = np.argmax(similarity, axis=None)
    
    print('Your question:', question)
    print('Closest question found:', all_data.iloc[max_similarity]['Question'])
    ans=['']
    print('Answer:', all_data.iloc[max_similarity]['Answer'])
    ans=('Answer:', all_data.iloc[max_similarity]['Answer'])

    label['text'] = ans
    
canvas=tkinter.Canvas(root,height=Height,width=Width)
canvas.pack()

root.title('Question Answer System')

frame = tkinter.Frame(root, bg='#80c1ff', bd=5)
frame.place(relx=0.5, rely=0.1, relwidth=0.75, relheight=0.1, anchor='n')
w = tkinter.Label(root, text="Question Answer System")
w.pack()
entry = tkinter.Entry(frame, font=40)
entry.place(relwidth=0.65, relheight=1)

button = tkinter.Button(frame, text="Get Answer", font=40,command=lambda:ask_question(entry.get()))
button.place(relx=0.7, relheight=1, relwidth=0.3)

lower_frame = tkinter.Frame(root, bg='#80c1ff', bd=10)
lower_frame.place(relx=0.5, rely=0.25, relwidth=0.75, relheight=0.6, anchor='n')

label = tkinter.Label(lower_frame)
label.config(font=("Courier", 24))
label.place(relwidth=1, relheight=1)

root.mainloop()
