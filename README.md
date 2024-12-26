# Project Rag Mistral

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system using Python. The main goal is to fetch an essay, split it into chunks, generate embeddings for these chunks, and then use these embeddings to answer a specific question about the essay. The project leverages various tools and libraries such as Mistral, FAISS, and dotenv.

## Project Structure

## Files and Directories

- **.env**: Contains environment variables such as API keys.
- **.gitignore**: Specifies files and directories to be ignored by Git.
- **Docker/**: Contains Docker-related files.
  - **dependency/**: Contains the `requirements.txt` file listing the Python dependencies.
- **Dockerfile**: Defines the Docker image for the project.
- **essay.txt**: The fetched essay text.
- **rag.py**: The main script that implements the RAG system.
- **utils/**: Contains utility scripts.
  - **batch_chunks.py**: Contains the `batch_chunks` function to split text into batches.

## What I Did

1. **Fetched an Essay**: Used the `requests` library to fetch an essay from a URL and saved it to a file.
2. **Split Text into Chunks**: Split the fetched essay into smaller chunks for processing.
3. **Generated Embeddings**: Used the Mistral API to generate embeddings for the text chunks.
4. **Implemented Batching**: Created a batching mechanism to ensure the token count per batch does not exceed a specified limit.
5. **Built a FAISS Index**: Used FAISS to create an index of the embeddings for efficient similarity search.
6. **Answered a Question**: Formulated a question, generated its embedding, and searched the FAISS index to retrieve relevant chunks.
7. **Generated a Response**: Used the Mistral API to generate a response based on the retrieved chunks.

## What I Solved

- **Token Limit Management**: Implemented a batching mechanism to manage token limits when generating embeddings.
- **Rate Limiting**: Added retry logic with exponential backoff to handle rate limits from the Mistral API.
- **Efficient Search**: Used FAISS to efficiently search for similar text chunks based on embeddings.

## What I Learned

- **RAG Systems**: Gained a deeper understanding of how Retrieval-Augmented Generation systems work.
- **API Integration**: Learned how to integrate and handle API responses, including error handling and rate limiting.
- **Text Embeddings**: Understood the process of generating and using text embeddings for similarity search.
- **Docker**: Learned how to containerize a Python application using Docker.
- **Batch Processing**: Implemented batch processing to handle large texts within token limits.

## How to Run

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/hadjuse/projectRagMistral.git
    cd projectRagMistral
    ```

2. **Set Up Environment Variables**:
    - Create a [.env](http://_vscodecontentref_/4) file with the necessary API keys.

3. **Build and Run Docker Container**:
    ```sh
    docker build -t rag-system .
    docker run -p 80:80 rag-system
    ```

4. **Check the Output**:
    - The script will fetch the essay, process it, and print the answer to the specified question.

## Dependencies

- Python 3.10
- Docker
- Libraries listed in [requirements.txt](http://_vscodecontentref_/5)

## Future Improvements

- **Error Handling**: Improve error handling for various edge cases.
- **Scalability**: Optimize the batching mechanism for larger texts.
- **User Interface**: Develop a simple web interface to interact with the RAG system.

## Acknowledgments

- **Mistral**: For providing the API to generate embeddings.
- **FAISS**: For the efficient similarity search library.
- **dotenv**: For managing environment variables.

## Contact

For any questions or suggestions, please contact [rabearimananahadjpro@gmail.com].

## Source
[Source documentation for Rag](https://docs.mistral.ai/guides/rag/)