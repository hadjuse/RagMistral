def batch_chunks(chunks, max_tokens=16000, token_per_char=0.25):
    """
    Splits the list of chunks into batches where the estimated token count per batch does not exceed max_tokens.
    """
    batches = []
    current_batch = []
    current_tokens = 0
    
    for chunk in chunks:
        # Estimate tokens for the chunk
        chunk_tokens = int(len(chunk) * token_per_char)
        
        if current_tokens + chunk_tokens > max_tokens:
            if current_batch:
                batches.append(current_batch)
            current_batch = [chunk]
            current_tokens = chunk_tokens
        else:
            current_batch.append(chunk)
            current_tokens += chunk_tokens
    
    if current_batch:
        batches.append(current_batch)
    
    return batches