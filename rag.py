import os
from mistralai import Mistral
import requests
import numpy as np
import faiss
from dotenv import load_dotenv
import time
import random
from utils.batch_chunks import batch_chunks
# Load environment variables
load_dotenv()
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("API key for RAG_API_KEY not found in environment variables")

# Initialize Mistral client
client = Mistral(api_key=api_key)

# Fetch the essay
response = requests.get('https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt')
text = response.text

# Save the essay to a file
with open('essay.txt', 'w') as f:
    f.write(text)

# Split text into chunks
chunk_size = 2048  # Adjust if necessary
chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
print(f"Number of chunks: {len(chunks)}")

def get_text_embeddings_in_batches(chunks, max_tokens=16000, token_per_char=0.25, max_retries=5, backoff_factor=2):
    """
    Retrieves embeddings for text chunks in controlled batches to respect token limits.
    """
    batches = batch_chunks(chunks, max_tokens, token_per_char)
    all_embeddings = []
    
    for idx, batch in enumerate(batches):
        for attempt in range(max_retries):
            try:
                print(f"Processing batch {idx + 1}/{len(batches)} with {len(batch)} chunks...")
                embeddings_batch_response = client.embeddings.create(
                    model="mistral-embed",
                    inputs=batch
                )
                batch_embeddings = [item.embedding for item in embeddings_batch_response.data]
                all_embeddings.extend(batch_embeddings)
                break  # Exit retry loop on success
            except Exception as e:
                if "rate limit" in str(e).lower():
                    sleep_time = backoff_factor ** attempt + random.uniform(0, 1)
                    print(f"Rate limit exceeded. Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                elif "Too many tokens in batch" in str(e):
                    # Further reduce batch size if token limit is exceeded
                    print("Batch token limit exceeded. Reducing batch size.")
                    # Split current batch into smaller batches
                    smaller_batches = batch_chunks(batch, max_tokens // 2, token_per_char)
                    # Insert smaller batches into the main batches list
                    batches = batches[:idx] + smaller_batches + batches[idx + 1:]
                    break  # Break retry loop to process smaller batches
                else:
                    print(f"Unexpected error: {e}")
                    raise e
        else:
            raise Exception("Max retries exceeded for embedding requests.")
    
    return np.array(all_embeddings)

# Fetch embeddings with batching
text_embeddings = get_text_embeddings_in_batches(chunks)
if text_embeddings.size == 0:
    raise Exception("Failed to fetch text embeddings.")

# Initialize FAISS index
vectorDatabase = faiss.IndexFlatIP(text_embeddings.shape[1])
vectorDatabase.add(text_embeddings)

# Prepare the question
question = "What were the two main things the author worked on before college?"

def get_question_embedding(question, max_tokens=16000, token_per_char=0.25, max_retries=5, backoff_factor=2):
    """
    Retrieves embedding for the question with retry logic.
    """
    for attempt in range(max_retries):
        try:
            print("Processing question embedding...")
            embeddings_batch_response = client.embeddings.create(
                model="mistral-embed",
                inputs=[question]
            )
            question_embedding = embeddings_batch_response.data[0].embedding
            return np.array([question_embedding])
        except Exception as e:
            if "rate limit" in str(e).lower():
                sleep_time = backoff_factor ** attempt + random.uniform(0, 1)
                print(f"Rate limit exceeded for question embedding. Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"Unexpected error: {e}")
                raise e
    raise Exception("Max retries exceeded for question embedding.")

# Get embedding for the question
question_embeddings = get_question_embedding(question)

# Search the FAISS index
D, I = vectorDatabase.search(question_embeddings, k=2)  # distance, index
retrieved_chunk = [chunks[i] for i in I.tolist()[0]]

# Construct the prompt
prompt = f"""
Context information is below.
---------------------
{retrieved_chunk}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {question}
Answer:
"""

def run_mistral(user_message, model="mistral-large-latest", max_retries=5, backoff_factor=2):
    """
    Runs the Mistral chat completion with retry logic for rate limits.
    """
    messages = [
        {
            "role": "user", "content": user_message
        }
    ]
    for attempt in range(max_retries):
        try:
            print("Sending chat completion request...")
            chat_response = client.chat.complete(
                model=model,
                messages=messages
            )
            return chat_response.choices[0].message.content
        except Exception as e:
            if "rate limit" in str(e).lower():
                sleep_time = backoff_factor ** attempt + random.uniform(0, 1)
                print(f"Rate limit exceeded for chat API. Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            elif "Too many tokens in batch" in str(e):
                print("Chat API prompt too long. Consider reducing the context size.")
                return "Sorry, the prompt is too long for processing."
            else:
                print(f"Unexpected error: {e}")
                return "Sorry, I couldn't process your request at this time."
    return "Sorry, I couldn't process your request at this time."

# Get the answer with retry logic
answer = run_mistral(prompt)
print("Answer:", answer)
