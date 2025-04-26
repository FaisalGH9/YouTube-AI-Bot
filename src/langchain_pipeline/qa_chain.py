from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain import PromptTemplate
from src.config.settings import OPENAI_API_KEY

def get_response_from_query(db, query, k=4, model_name="gpt-3.5-turbo-instruct"):
    """
    Get a response from the vector database based on a query
    
    Args:
        db: The Chroma vector database
        query: The user's question
        k: Number of chunks to retrieve
        model_name: The OpenAI model to use
    
    Returns:
        A response to the question
    """
    # For very specific queries, use similarity search with relevance scores
    docs = db.similarity_search_with_relevance_scores(query, k=k*2)
    
    # Filter to docs with relevance above threshold
    threshold = 0.7
    relevant_docs = [doc for doc, score in docs if score > threshold]
    
    # If we have too few relevant docs, fall back to regular similarity search
    if len(relevant_docs) < k:
        relevant_docs = [doc for doc, _ in docs[:k]]
    else:
        # Keep only the k most relevant
        relevant_docs = relevant_docs[:k]
    
    # TOKEN LIMIT HANDLING: Calculate estimated tokens for context
    # Conservative estimate - 1 token ≈ 4 chars
    context_text = " ".join([d.page_content for d in relevant_docs])
    estimated_tokens = len(context_text) // 4
    
    # If estimated tokens exceed limits, reduce context intelligently
    max_context_tokens = 3000  # Leave room for prompt and completion
    
    if estimated_tokens > max_context_tokens:
        # Try with fewer documents first
        if len(relevant_docs) > 2:
            # Just use the 2 most relevant docs
            relevant_docs = relevant_docs[:2]
            context_text = " ".join([d.page_content for d in relevant_docs])
            estimated_tokens = len(context_text) // 4
    
    # If still too large, truncate each document
    if estimated_tokens > max_context_tokens:
        # Calculate how much we need to trim
        total_chars = len(context_text)
        target_chars = max_context_tokens * 4
        ratio = target_chars / total_chars
        
        # Truncate each document proportionally
        truncated_docs = []
        for doc in relevant_docs:
            content_len = len(doc.page_content)
            # Keep at least the first 20% of each document
            keep_chars = max(int(content_len * ratio), int(content_len * 0.2))
            truncated_content = doc.page_content[:keep_chars]
            truncated_docs.append(truncated_content)
            
        context_text = " ".join(truncated_docs)
    
    # Use model-specific settings
    if "gpt-4" in model_name:
        temperature = 0.1
        max_tokens = 750
    else:
        temperature = 0
        max_tokens = 500
    
    llm = OpenAI(
        model=model_name, 
        temperature=temperature, 
        max_tokens=max_tokens,
        openai_api_key=OPENAI_API_KEY
    )
    
    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant that can answer questions about YouTube videos 
        based on the video's transcript.

        Answer the following question: {question}
        By searching the following video transcript: {docs}

        Only use the factual information from the transcript to answer the question.
        If you feel like you don't have enough information to answer the question, say "I don't know".
        Your answers should be detailed but concise.
        """
    )
    
    # Final check - if still too large, use an even more aggressive approach
    if len(context_text) // 4 > max_context_tokens:
        # Extreme fallback - just use small excerpts from each document
        excerpts = []
        for doc in relevant_docs:
            # Take first 300 chars of each document
            excerpts.append(doc.page_content[:300])
        context_text = " ".join(excerpts)
    
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(question=query, docs=context_text)

def get_simple_response_from_query(db, query, model_name="gpt-3.5-turbo-instruct"):
    """
    Simplified version for very long videos - uses minimal context
    """
    # Get just 2 most relevant chunks
    docs = db.similarity_search(query, k=2)
    
    # Extract very short snippets
    snippets = []
    for doc in docs:
        # Find sentences that might contain the answer
        content = doc.page_content.lower()
        query_terms = query.lower().split()
        
        # Find a relevant excerpt
        best_pos = 0
        max_matches = 0
        
        # Simple sliding window to find most relevant section
        for i in range(0, len(content), 50):
            window = content[i:i+300]
            matches = sum(1 for term in query_terms if term in window)
            if matches > max_matches:
                max_matches = matches
                best_pos = i
        
        # Take excerpt around the best position
        start_pos = max(0, best_pos - 50)
        excerpt = doc.page_content[start_pos:start_pos+250]
        snippets.append(excerpt)
    
    context = " ".join(snippets)
    
    llm = OpenAI(
        model=model_name, 
        temperature=0, 
        max_tokens=300,
        openai_api_key=OPENAI_API_KEY
    )
    
    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        Answer this question briefly: {question}
        Based on these transcript excerpts: {docs}
        Keep your answer concise and factual.
        """
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(question=query, docs=context)