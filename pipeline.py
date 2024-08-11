import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import sys
import ast
import os
import time
import re

# FIX ERROR: pip freeze | grep aijson
# allow sys.argv[1] when calling from main.py

from aijson import Flow
import asyncio

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForTokenClassification, pipeline
from sentence_transformers import SentenceTransformer

PROMPT = open('prompt.txt', 'r').read().strip()

# Environment variables

# To remove API call errors
DELAY = 10  # seconds

# Configuration
N_LARGEST = 3  # Number of top results to display
MAX_CHARS = 200  # Length of characters beyond which we summarize

# Model placeholders
sentence_model = None
summarizer = None
ner_pipeline = None
global_tokenizer = None

# Similarity extraction
def get_sentence_model():
    global sentence_model
    if sentence_model is None:
        sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return sentence_model

# Lazy loading of the summarizer
def get_summarizer():
    global summarizer
    if summarizer is None:
        global global_tokenizer
        if global_tokenizer is None:
            global_tokenizer = AutoTokenizer.from_pretrained("edncodeismad/finetuned_T5")
        model = AutoModelForSeq2SeqLM.from_pretrained("edncodeismad/finetuned_T5")
        summarizer = pipeline("summarization", model=model, tokenizer=global_tokenizer)
    return summarizer

# Lazy loading of the NER pipeline
def get_ner_pipeline():
    global ner_pipeline
    if ner_pipeline is None:
        ner_model = "dbmdz/bert-large-cased-finetuned-conll03-english"
        tokenizer = AutoTokenizer.from_pretrained(ner_model)
        model = AutoModelForTokenClassification.from_pretrained(ner_model)
        ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)
    return ner_pipeline

# Initialize models
sentence_model = get_sentence_model()
summarizer = get_summarizer()
ner_pipeline = get_ner_pipeline()

# Function to get large entries from DataFrame
def get_large_entries(df):
    df = df.to_numpy()
    long_idxs = []
    for i, row in enumerate(df):
        for j, el in enumerate(row):
            if len(str(el)) > MAX_CHARS:
                long_idxs.append((i, j))
    return long_idxs

# Function to get scores based on cosine similarity
def get_scores(arr, query):
    query_emb = sentence_model.encode(query)
    scores = np.zeros_like(arr)

    # Iterate over each element in the 1D array
    for i, el in enumerate(arr):
        el_emb = sentence_model.encode([str(el)])
        query_tensor = torch.tensor(query_emb)  # Ensure tensor is on CPU
        el_tensor = torch.tensor(el_emb)  # Ensure tensor is on CPU
        simil = F.cosine_similarity(query_tensor, el_tensor)
        scores[i] = simil.item()

    # Find the indices of the N_LARGEST scores
    idx = np.argpartition(scores, -N_LARGEST)[-N_LARGEST:]
    idx = np.flip(idx)  # To get the indices in descending order of score

    return idx.tolist()

# Function to extract array from text
def extract_array_from_text(text):
    pattern = r'\[.*?\]'
    match = re.search(pattern, text)
    if match:
        array_str = match.group(0)
        try:
            array = ast.literal_eval(array_str)
            return array
        except (ValueError, SyntaxError):
            print("Error: The extracted content is not a valid Python array.")
            return []
    else:
        print("Error: No array found in the input text.")
        return []

# Function to handle retries with exponential backoff
async def make_api_call_with_retries(flow, max_retries=5, base_delay=1.0):
    retries = 0
    while retries < max_retries:
        try:
            result = await flow.run()
            return result
        except litellm.exceptions.RateLimitError as e:
            delay = base_delay * (2 ** retries) + random.uniform(0, 1)
            print(f"Rate limit exceeded. Retrying in {delay:.2f} seconds...")
            time.sleep(delay)
            retries += 1

    raise Exception("Max retries exceeded. Could not complete the API call.")

# Function to get NER summary
async def get_ner_summary(prompt):
    flow = Flow.from_file('get_nes.ai.yaml')
    flow = flow.set_vars(prompt=prompt)
    result = await make_api_call_with_retries(flow)
    return result

# Function to extract named entities
async def get_ners(string, wait_time=1.0):
    ner_output = ner_pipeline(string)

    entities = {}
    ners = []

    for entity in ner_output:
        entity_text = entity['word'].replace("##", "")
        entity_label = entity['entity']

        if entity_label not in entities:
            entities[entity_label] = []

        entities[entity_label].append(entity_text)

    async def gather_results():
        tasks = []
        for PROMPT in entities.values():
            tasks.append(get_ner_summary(PROMPT))
        return await asyncio.gather(*tasks)

    result = await gather_results()  # Use await instead of asyncio.run()
    for res in result:
        res = extract_array_from_text(res)  # Safely extract the array from the text
        if res:
            ners.extend(res)

    return ' '.join(ners)

# Function to extract NER from DataFrame
async def get_ners_from_df(df_arg, wait_time=1.0):
    ners = []
    df = df_arg.copy()

    # Combine all the row entries into one input
    df['inputs'] = df.apply(lambda row: ', '.join(row.values.astype(str)), axis=1)

    for input in df['inputs']:
        ner = await get_ners(input, wait_time=wait_time)  # Use await instead of just calling the function
        ners.append(ner)
        time.sleep(wait_time)  # Wait between each NER extraction

    return ners

async def get_data(df, query):
    long_idxs = get_large_entries(df)
    df_summarized = df.copy()
    for idx in long_idxs:
        long_txt = df.iloc[idx[0], idx[1]]
        summary = summarizer(long_txt)
        df_summarized.iloc[idx[0], idx[1]] = summary

    NERs = await get_ners_from_df(df)  # Await the asynchronous function

    assert len(NERs) == len(df), 'Numbers of NERs should match size of database'
    df_summarized['NER'] = NERs

    person_array = df.apply(lambda row: ''.join(row.astype(str)), axis=1)
    query_scores = get_scores(person_array.to_numpy(), query)

    df_summarized.to_csv('df_summarized.csv')
    return person_array, query_scores

async def run_get_reason(query, entry):
    flow = Flow.from_file('get_reason.ai.yaml')
    flow = flow.set_vars(query=query, entry=entry)
    result = await flow.run()
    return result

async def get_reason(query, entry):
    result = await run_get_reason(query, entry)
    result = result.strip()
    return result

async def find_relevant_data(data, prompt):  # data is a dictionary, prompt is a string
    df = pd.DataFrame(data)
    names = [entry for entry in data['Name']]

    output = pd.DataFrame(columns=['Name', 'Reason'])

    person_array, query_scores = await get_data(df, prompt)
    person_array = person_array.to_numpy()

    ranked_entries = person_array[np.array(query_scores)]

    for i, entry in enumerate(ranked_entries):
        reason = await get_reason(prompt, entry)
        new_entry = np.array([names[i], str(reason)])
        output.loc[len(output)] = new_entry

    return output

"""EXAMPLE RUN"""

import litellm
import aijson

data = pd.read_csv('sample.csv')

async def get_output():
    output = await find_relevant_data(data, PROMPT)
    return output

#PROMPT = 'Who is an astrophysicist?'

if __name__ == '__main__':
    OUTPUT_DICT = asyncio.run(get_output())
    OUTPUT_DICT.to_csv('OUTPUT.csv')
    
