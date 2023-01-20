import re
import os
import openai
from time import time, sleep
import textwrap
import pandas as pd
import ast
import requests
import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity
import tiktoken
import json
import bs4 as bs
import time




def open_file(filepath):
    # This function opens a file located at the specified filepath and returns a string containing the file's content.
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def save_file(filepath, content):
    # This function saves the content in a file located at the specified filepath.
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

# change path to where you store your credentials
openai.api_key = open_file('creds/creds.txt')

rapid_api_key = open_file('creds/rapid_api_key.txt')
rapid_api_host = open_file('creds/rapid_api_host.txt')

def gpt3_completion(prompt, label='gpt3', engine='text-davinci-003', temp=0, top_p=1.0, tokens=400, freq_pen=2.0, pres_pen=2.0, stop=['asdfasdf', 'asdasdf']):
    # This function uses OpenAI's GPT-3 Engine to generate completions for the given prompt.
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII', errors='ignore').decode()  # force it to fix any unicode errors
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop)
            text = response['choices'][0]['text'].strip()
            text = re.sub('\s+', ' ', text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)
            
def deduplicate_list(x):
    # This function removes duplicate elements from a list and returns the deduplicated list.
    return list(dict.fromkeys(x))

def get_search_strings(input):
    ### input is a string, containing the initial question of the user
    ### Returns string containing a JSON formatted response
    question = input
    prompt = """The following is a question a user has.  Please propose a list of between 5 and 10 potentially useful Google Searches the user can execute to retrieve useful information to answer the question.  The list should be output as a python list, using double quotes for strings.
    QUESTION:
    {0}

    LIST RESPONSE: """.format(question)
    gpt_response = gpt3_completion(prompt)
    try:
        search_list = ast.literal_eval(gpt_response)
    except:
        gpt_response = gpt3_completion(prompt, temp=0.6)
        search_list = ast.literal_eval(gpt_response)
    return search_list

def get_top_urls(query, n, pprint = True):
    # input is a google search (a string) and an integer n
    # output top n urls
    url = "https://google-search72.p.rapidapi.com/search"
    #num is number n of top results
    querystring = {"query":query,"gl":"us","lr":"en","num":str(n),"start":"0","sort":"relevance"}

    headers = {
        "X-RapidAPI-Key": rapid_api_key,
        "X-RapidAPI-Host": rapid_api_host
    }

    response = requests.request("GET", url, headers=headers, params=querystring)
    if pprint:
        global resp_debug 
        resp_debug = response.text
    data = json.loads(response.text)
    # Get list of top urls
    top_urls = []
    for item in data['items']:
        if not item['link'].endswith(".pdf"):
            top_urls.append(item['link'])
        
    return top_urls

# Functions to fetch the text
def get_website(url):
    # The input is a string containing the desired url
    # The output is a string containing the text from the website at the url
    # The function uses the requests library to get the text from the website
    # at the url and returns the text
    try:
        response = requests.get(url)
        return response.text
    except:
        return 'Error'

def extract_clean_text(site):
    # The input is a string containing all the text from a website, including the HTML
    # The output is a string containing only the human readable text i.e. without all the HTML tags
    # The function uses the BeautifulSoup library to parse the HTML and return only the human readable text
    # Parse the HTML as a string
    soup = bs.BeautifulSoup(site,'html.parser')
    # Get the text out of the soup and return it
    text = ''.join(map(lambda p: p.text, soup.find_all('p')))
    return text

def split_string(string, x = 2500): 
    # Split the string into n-sized chunks. Last chunk might be smol.
    res=[string[y-x:y] for y in range(x, len(string)+x,x)]
    #split_strings.append(res) 
    return res

def create_df(top_urls, pprint = True):
    # Input is a deduplicated list of urls
    # Output is a dataframe containing url, snippet_id, text, embedding obtained by fetching the website and calling openai
    df = pd.DataFrame({'url': pd.Series(dtype='str'),
                   'snippet_id': pd.Series(dtype='str'),
                   'text': pd.Series(dtype='str'),
                  'embedding': pd.Series(dtype='str')})
    # Using NumPy
    dtypes = np.dtype(
        [
            ("url", str),
            ("snippet_id", str),
            ("text", str),
            ("embedding",str)
        ]
    )
    df = pd.DataFrame(np.empty(0, dtype=dtypes))
    all_df_list = []
    embedding_model = "text-embedding-ada-002"
    embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
    max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191
    encoding = tiktoken.get_encoding(embedding_encoding)
    for url in top_urls:
        text = get_website(url)
        if pprint:
            print(url)
        if text =='Error':
            continue
        try:
            clean = extract_clean_text(text)
        except Exception as e:
            print(e)
            continue
        snippet_list = split_string(clean, 2500)
        #url_list = [url] * len(snippet_list)   
        if pprint:
            print('running encoder')
        encoding_list = [get_embedding(snippet, engine=embedding_model) for snippet in snippet_list]
        if pprint:
            print('finished encoder')
            print('adppending to dataframe')
        for i in range(len(snippet_list)):
            all_df_list.append([url, i, snippet_list[i], encoding_list[i]])
            #print(all_df_list[i][0])
            #print(all_df_list[i][1])
            #df = pd.concat([df,pd.DataFrame({'url': [url], 'snippet_id': [i], 'text':[snippet_list[i]], 'embedding': [encoding_list[i]]})], ignore_index=True) 
        all_df = pd.concat([df,pd.DataFrame(all_df_list, columns=df.columns)], ignore_index=True)       
    return all_df 


# search through the reviews for a specific product
def search_snippets(df, question, n=3, pprint=False):
    # input is a df with embeddings of the texts (it contains url, snippet_id, snippet text as well)
    # output is the top n results, in the same format as df
    question_embedding = get_embedding(
        question,
        engine="text-embedding-ada-002"
    )
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, question_embedding))

    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)
    )
    if pprint:
        for r in results:
            print(r[:200])
            print()
    return results

def create_df_urls(top_urls, df, my_question, pprint = True):
    """This function creates a dataframe (df_urls) of URLs and their respective trust scores.
    Inputs: 
    top_urls (list): A list of URLs
    df (DataFrame): A DataFrame
    my_question (str): A string representing a question
    pprint (bool): A boolean determining if print statements should be executed
    Outputs:
    df_urls (DataFrame): A DataFrame of URLs and their respective trust scores
    """
    
    df_urls = pd.DataFrame({'url': pd.Series(dtype='str'),
                       'score': pd.Series(dtype='int')})
    score_list = []
    for url in top_urls:
        prompt_template = """Assume you are a University Profressor, experienced in researching and evaluating sources of information. 
        Here is my question. I google "{0}" and got this link: {1} Can I trust the information on this url? 
        From 1 to 10, how would you score the reputation of this source? Please only output the number.
        """
        prompt = prompt_template.format(my_question, url)
        #print(prompt)
        if pprint:
            print(url)
            print('fetch gpt')
        gpt_response = gpt3_completion(prompt)
        if pprint:
            print(gpt_response)
        temp =0
        # If we don't get a numeric response or number out of 1-10 range, keep trying with more temperature
        for i in range(4):
            if gpt_response.isnumeric()==True:
                if int(gpt_response) in range(11):
                    score_list.append(int(gpt_response))
                    df_urls = pd.concat([df_urls,pd.DataFrame({'url': [url],'score': [int(gpt_response)]})], ignore_index=True) 
                    break
                else:
                    temp = temp + 2*i/10
                    if pprint:
                        print('fetch gpt on rebound')
                    gpt_response = gpt3_completion(prompt, temp=temp)
                    if pprint:
                        print(gpt_response)
                    counter = counter + 1
                    continue
            else:
                temp = temp + 2*i/10
                gpt_response = gpt3_completion(prompt, temp=temp)
    return df_urls
    

def find_snippets(my_question, df, df_urls, top_m_urls, top_n_snippets):
    """This function takes a question, a dataframe, a dataframe of urls, the number of top URLs, and the number of top snippets 
        as input, and returns a dataframe of the most relevant snippets as output. It first sorts the urls dataframe by score
        and takes the top m urls. It then merges this dataframe with the original dataframe and removes any duplicated columns. 
        It then searches the dataframe for the most relevant snippets and returns a dataframe of the top n snippets.
    """
    url_sorted = df_urls.sort_values(by='score', ascending=False).head(top_m_urls)
    df = pd.merge(df,url_sorted, on='url', how='inner')
    df = df.loc[:,~df.columns.duplicated()].copy()
    df = search_snippets(df, my_question, n = top_n_snippets, pprint=True)
    return df

def get_answer(search_df, my_question, pprint = True):
    """
    This function takes three parameters: a search dataframe consisting of text, a question string, and a 
    boolean value for pprint. It uses a GPT-3 completion to extract an answer from each row of the search dataframe based 
    on the question string, appending the answer to the search dataframe as a new column. If the pprint parameter is set to 
    True, the prompt and answer will be printed to the console. The function returns the search 
    dataframe with the answers added.
    """
    gpt_answers = []
    for index, row in search_df.iterrows():
        prompt = """I have a question and a paragraph.  Please extract an answer from the paragraph if present, otherwise say "No relevant information here
        QUESTION:
        {0}
        PARAGRAPH:
        {1}
        ANSWER:""".format(my_question, row['text'])
        completion = gpt3_completion(prompt)
        gpt_answers.append(completion)
        if pprint:
            print(prompt)
    search_df['answer'] = gpt_answers
    return search_df


# a function to run all of the above process
def run_search(my_question, urls_per_search, top_m_urls, top_n_snippets, pprint = True):
    """This function runs a search for a given question and returns the top search results and snippets, as well as 
    proposed answer by GPT-3. It takes in 4 parameters: my_question (the question to be searched for), urls_per_search
    (the number of urls per search), top_m_urls (the top number of urls to be returned), and top_n_snippets 
    (the top number of snippets to be returned). The output is a search_df dataframe that contains the top search 
    results and snippets for the given question.
    """
    search_strings = get_search_strings(my_question)
    print(search_strings)
    top_urls = []
    for search in search_strings:
        top_urls_search = get_top_urls(search, urls_per_search, pprint)
        top_urls = top_urls_search + top_urls
        top_urls = deduplicate_list(top_urls)
        sleep(1)
    print('total urls: ')
    print(len(top_urls))
    df = create_df(top_urls, pprint)
    #global z # Yes, this is horrible, but helped me debug..
    #z = df
    print('df created with length:')
    print(len(df))
    df_urls = create_df_urls(top_urls, df, my_question, pprint)
    #global a 
    #a = df_urls
    print('df_urls done')
    search_df = find_snippets(my_question, df, df_urls, top_m_urls, top_n_snippets)
    #global b
    #b = search_df
    print('search_df-done')
    search_df = get_answer(search_df, my_question, pprint = pprint)
    return search_df





