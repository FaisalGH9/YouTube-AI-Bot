from langchain.smith import RunEvalConfig
from langchain.evaluation.criteria import LLMCriteriaEvaluator
from langchain.smith import traceable
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain import PromptTemplate
from langchain.llms import OpenAI
from typing import List, Dict, Any, Optional, Union
from src.config.settings import (
    OPENAI_API_KEY, 
    LANGSMITH_API_KEY, 
    LANGSMITH_PROJECT_NAME, 
    LANGSMITH_TRACING_ENABLED
)
import os

# Configure LangSmith tracing
if LANGSMITH_TRACING_ENABLED and LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_TRACING"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT_NAME
else:
    os.environ["LANGCHAIN_TRACING"] = "false"

class EvaluationService:
    """Handles evaluation of AI responses using LangSmith"""
    
    def __init__(self):
        self.eval_llm = OpenAI(model_name="gpt-4", temperature=0, openai_api_key=OPENAI_API_KEY)
    
    @traceable(name="qa_chain")
    def traceable_qa(self, chain_func, *args, **kwargs):
        """Wraps QA function with LangSmith tracing"""
        return chain_func(*args, **kwargs)
    
    @traceable(name="summarization_chain")
    def traceable_summarize(self, chain_func, *args, **kwargs):
        """Wraps summarization function with LangSmith tracing"""
        return chain_func(*args, **kwargs)
    
    @traceable(name="multimodal_chain")
    def traceable_multimodal(self, chain_func, *args, **kwargs):
        """Wraps multimodal function with LangSmith tracing"""
        return chain_func(*args, **kwargs)
    
    def create_qa_evaluator(self) -> LLMCriteriaEvaluator:
        """Create an evaluator for QA responses"""
        criteria = {
            "relevance": "The response directly addresses the question asked.",
            "accuracy": "The response only contains information from the video transcript.",
            "completeness": "The response thoroughly answers all aspects of the question.",
            "coherence": "The response is well-structured, logical, and easy to understand."
        }
        
        return LLMCriteriaEvaluator(
            criteria=criteria,
            llm=self.eval_llm,
            normalize_scores=True
        )
    
    def create_summary_evaluator(self) -> LLMCriteriaEvaluator:
        """Create an evaluator for summary responses"""
        criteria = {
            "conciseness": "The summary captures the essential points without unnecessary details.",
            "comprehensiveness": "The summary covers all the important topics from the video.",
            "accuracy": "The summary only contains information from the video.",
            "coherence": "The summary flows logically and is well-structured."
        }
        
        return LLMCriteriaEvaluator(
            criteria=criteria,
            llm=self.eval_llm,
            normalize_scores=True
        )
    
    def create_multimodal_evaluator(self) -> LLMCriteriaEvaluator:
        """Create an evaluator for multimodal responses"""
        criteria = {
            "visual_integration": "The response effectively incorporates visual information from the video.",
            "audio_visual_alignment": "The response correctly aligns spoken content with visual elements.",
            "timestamp_accuracy": "Any timestamps mentioned correspond to relevant content in the video.",
            "multimodal_reasoning": "The response shows understanding that integrates both audio and visual content."
        }
        
        return LLMCriteriaEvaluator(
            criteria=criteria,
            llm=self.eval_llm,
            normalize_scores=True
        )
    
    def evaluate_qa(self, question: str, response: str, ground_truth: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate a QA response
        
        Args:
            question: The user's question
            response: The system's response
            ground_truth: Optional reference answer (if available)
            
        Returns:
            Evaluation results as a dictionary
        """
        evaluator = self.create_qa_evaluator()
        
        # Prepare the evaluation inputs
        eval_input = {
            "input": question,
            "prediction": response
        }
        
        if ground_truth:
            eval_input["reference"] = ground_truth
            
        # Run evaluation
        result = evaluator.evaluate_strings(**eval_input)
        return result
    
    def evaluate_summary(self, transcript: str, summary: str) -> Dict[str, Any]:
        """
        Evaluate a summary response
        
        Args:
            transcript: The original transcript (or a representative sample)
            summary: The generated summary
            
        Returns:
            Evaluation results as a dictionary
        """
        evaluator = self.create_summary_evaluator()
        
        # Use a sample of the transcript if it's too long
        if len(transcript) > 10000:
            # Take evenly distributed samples
            parts = []
            step = len(transcript) // 10
            for i in range(0, len(transcript), step):
                parts.append(transcript[i:i+1000])
            transcript_sample = "\n...\n".join(parts)
        else:
            transcript_sample = transcript
        
        # Run evaluation
        result = evaluator.evaluate_strings(
            input=transcript_sample,
            prediction=summary
        )
        return result
    
    def evaluate_multimodal(self, query: str, response: str, 
                          transcript_sample: str, visual_descriptions: List[str]) -> Dict[str, Any]:
        """
        Evaluate a multimodal response
        
        Args:
            query: The user's question or request
            response: The system's response
            transcript_sample: A sample of the transcript
            visual_descriptions: List of visual frame descriptions
            
        Returns:
            Evaluation results as a dictionary
        """
        evaluator = self.create_multimodal_evaluator()
        
        # Create a combined input with both audio and visual information
        combined_input = {
            "query": query,
            "transcript_sample": transcript_sample,
            "visual_descriptions": visual_descriptions
        }
        
        # Convert to string format
        input_str = f"""
        QUERY: {query}
        
        TRANSCRIPT SAMPLE:
        {transcript_sample}
        
        VISUAL DESCRIPTIONS:
        {'. '.join(visual_descriptions)}
        """
        
        # Run evaluation
        result = evaluator.evaluate_strings(
            input=input_str,
            prediction=response
        )
        return result
    
    def run_eval_config(self, chain, dataset, eval_config=None):
        """
        Run a LangSmith evaluation on a chain with a dataset
        
        Args:
            chain: The LangChain chain to evaluate
            dataset: The evaluation dataset (list of examples)
            eval_config: Optional evaluation configuration
            
        Returns:
            Evaluation results
        """
        if not LANGSMITH_API_KEY or not LANGSMITH_TRACING_ENABLED:
            return {"error": "LangSmith evaluation requires API key and tracing enabled"}
        
        # Create default eval config if none provided
        if eval_config is None:
            eval_config = RunEvalConfig(
                evaluators=[
                    self.create_qa_evaluator(),
                    "embedding_distance"
                ]
            )
        
        # Run evaluation
        from langchain.smith import run_on_dataset
        
        eval_results = run_on_dataset(
            client=None,  # Use the client from environment variables
            dataset_name=f"{LANGSMITH_PROJECT_NAME}_eval",
            llm_or_chain_factory=chain,
            data=dataset,
            evaluation=eval_config,
            project_name=f"{LANGSMITH_PROJECT_NAME}_eval_results"
        )
        
        return eval_results