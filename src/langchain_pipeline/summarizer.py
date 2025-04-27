from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain import PromptTemplate
from typing import List, Dict, Any, Optional
from src.config.settings import OPENAI_API_KEY, DEFAULT_SUMMARY_MODEL

class VideoSummarizer:
    """Handles summarization of video content"""
    
    def summarize(self, db, model_name: str = DEFAULT_SUMMARY_MODEL, 
                summary_length: str = "Moderate") -> str:
        """
        Summarize video content from the vector database
        
        Args:
            db: The Chroma vector database
            model_name: The OpenAI model to use
            summary_length: 'Brief', 'Moderate', or 'Detailed'
        
        Returns:
            A summary of the video content
        """
        # Determine how many chunks to retrieve based on desired length
        if summary_length == "Brief":
            k = 5
            max_tokens = 100
        elif summary_length == "Detailed":
            k = 15
            max_tokens = 750
        else:  # Moderate
            k = 10
            max_tokens = 250
            
        # Check if we need to use a map-reduce approach (many chunks)
        total_chunks = len(db.get()['ids']) if hasattr(db, 'get') and callable(db.get) else 100
        
        if total_chunks > 20:
            return self._map_reduce_summarize(db, model_name, max_tokens, summary_length)
        else:
            # Standard approach for shorter videos
            all_docs = db.similarity_search("summary", k=k)
            context = " ".join([d.page_content for d in all_docs])

            # Check if context is too large (conservative estimate - 1 token ≈ 4 chars)
            estimated_tokens = len(context) // 4
            if estimated_tokens > 3000:  # Leave buffer for prompt and completion
                return self._map_reduce_summarize(db, model_name, max_tokens, summary_length)

            llm = OpenAI(model=model_name, temperature=0.3, max_tokens=max_tokens, openai_api_key=OPENAI_API_KEY)
            prompt = PromptTemplate(
                input_variables=["docs"],
                template="""
                Summarize the following video transcript in a clear, concise way. Focus on the main ideas, important moments, and relevant discussion points.

                Transcript:
                {docs}

                Summary:
                """
            )

            chain = LLMChain(llm=llm, prompt=prompt)
            summary = chain.run(docs=context)
            return " ".join(summary.split())

    def _map_reduce_summarize(self, db, model_name: str, max_tokens: int, summary_length: str) -> str:
        """
        Map-reduce approach for summarizing very long videos:
        1. Get several sample chunks from throughout the video
        2. Summarize each chunk independently
        3. Combine those summaries into an overall summary
        
        Args:
            db: The Chroma vector database
            model_name: The OpenAI model to use
            max_tokens: Maximum tokens for the final summary
            summary_length: Desired summary length
            
        Returns:
            A summary of the video content
        """
        # Set sample size based on summary length
        if summary_length == "Brief":
            sample_size = 10  # Fewer samples for brief summary
        elif summary_length == "Detailed":
            sample_size = 20  # More samples for detailed summary
        else:  # Moderate
            sample_size = 15
        
        # Determine total chunks
        total_chunks = len(db.get()['ids']) if hasattr(db, 'get') and callable(db.get) else 100
        
        # Calculate positions to sample (evenly distributed)
        sample_positions = []
        if total_chunks <= sample_size:
            sample_positions = list(range(total_chunks))
        else:
            # Take evenly spaced samples
            step = max(1, total_chunks // sample_size)
            sample_positions = list(range(0, total_chunks, step))[:sample_size]
        
        # Collect documents from these positions
        sampled_docs = []
        for i, pos in enumerate(sample_positions):
            query = f"position:{pos}"  # Just a placeholder query
            # Use i as a fallback in case pos is out of range
            results = db.similarity_search(query, k=1)
            if results:
                sampled_docs.append(results[0])
        
        # Ensure we have at least some documents
        if not sampled_docs and total_chunks > 0:
            # Fallback to just getting some documents with a generic query
            sampled_docs = db.similarity_search("summary", k=min(sample_size, total_chunks))
        
        # Initialize OpenAI client with conservative token limits
        llm = OpenAI(model=model_name, temperature=0.3, max_tokens=100, openai_api_key=OPENAI_API_KEY)
        
        # Map: Summarize each chunk with very tight token constraints
        chunk_summaries = []
        chunk_prompt = PromptTemplate(
            input_variables=["docs"],
            template="""
            Briefly summarize this section of a video transcript in 2-3 sentences:
            {docs}
            
            Very brief summary:
            """
        )
        
        chunk_chain = LLMChain(llm=llm, prompt=chunk_prompt)
        
        # Process each document with error handling
        for doc in sampled_docs:
            try:
                # Truncate document if it's too large (approximate 3000 token limit ≈ 12000 chars)
                content = doc.page_content
                if len(content) > 12000:
                    content = content[:12000] + "..."
                    
                summary = chunk_chain.run(docs=content)
                chunk_summaries.append(summary.strip())
            except Exception as e:
                # If we hit an error, use a shorter chunk and try again
                try:
                    shorter_content = doc.page_content[:6000] + "..."
                    summary = chunk_chain.run(docs=shorter_content)
                    chunk_summaries.append(summary.strip())
                except:
                    # If still failing, just skip this chunk
                    continue
        
        # If we have too many summaries, combine them in batches
        if len(chunk_summaries) > 20:
            # Combine into batches of 5
            batch_size = 5
            batched_summaries = []
            for i in range(0, len(chunk_summaries), batch_size):
                batch = chunk_summaries[i:i+batch_size]
                batched_summaries.append(" ".join(batch))
            chunk_summaries = batched_summaries
        
        # Reduce: Combine the summaries with appropriate token limits
        combined_context = " ".join(chunk_summaries)
        
        # Final summarization
        reduce_prompt = PromptTemplate(
            input_variables=["summaries"],
            template="""
            Below are summaries from different parts of a video. Create a coherent overall summary 
            that captures the main points and narrative of the entire video:
            
            {summaries}
            
            Overall video summary:
            """
        )
        
        reduce_chain = LLMChain(
            llm=OpenAI(model=model_name, temperature=0.3, max_tokens=max_tokens, openai_api_key=OPENAI_API_KEY),
            prompt=reduce_prompt
        )
        
        # Handle potential token limit for final reduction
        if len(combined_context) > 12000:  # Conservative limit (approx 3000 tokens)
            # Further reduce by taking only the beginning, middle and end of summaries
            third = len(chunk_summaries) // 3
            selected_summaries = (
                chunk_summaries[:third] + 
                chunk_summaries[third:2*third][:3] + 
                chunk_summaries[2*third:]
            )
            combined_context = " ".join(selected_summaries)
        
        # Final check for length
        if len(combined_context) > 12000:
            combined_context = combined_context[:12000]
        
        try:
            final_summary = reduce_chain.run(summaries=combined_context)
            return final_summary.strip()
        except Exception as e:
            # Fallback with even shorter content if we still hit token limits
            return f"This video is extremely long and contains too much content for a complete summary. Here are key points from parts of the video: {combined_context[:2000]}..."