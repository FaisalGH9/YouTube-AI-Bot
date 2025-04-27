import os
import json
import openai
from langdetect import detect, LangDetectException
import iso639
from typing import Dict, Any, List, Optional, Tuple
from src.config.settings import OPENAI_API_KEY

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

class LanguageProcessor:
    """Handles language detection and translation for multilingual support"""
    
    # Mapping of language codes to full names
    LANGUAGE_MAP = {
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'ja': 'Japanese',
        'ko': 'Korean',
        'zh': 'Chinese',
        'ar': 'Arabic',
        'hi': 'Hindi',
        'nl': 'Dutch',
        'sv': 'Swedish',
        'fi': 'Finnish',
        'no': 'Norwegian',
        'da': 'Danish',
        'pl': 'Polish',
        'tr': 'Turkish',
        'vi': 'Vietnamese',
        'th': 'Thai'
    }
    
    @staticmethod
    def detect_language(text: str) -> Tuple[str, str]:
        """
        Detect the language of a text
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (language code, language name)
        """
        try:
            # Get a sample of the text (first 1000 chars)
            sample = text[:1000]
            
            # Detect language code
            lang_code = detect(sample)
            
            # Convert to language name
            try:
                lang_name = iso639.languages.get(part1=lang_code).name
            except:
                # Fallback to our map
                lang_name = LanguageProcessor.LANGUAGE_MAP.get(lang_code, 'Unknown')
            
            return lang_code, lang_name
            
        except LangDetectException:
            return 'unknown', 'Unknown'
    
    @staticmethod
    def translate_text(text: str, target_language: str = 'en') -> str:
        """
        Translate text to the target language using OpenAI
        
        Args:
            text: Text to translate
            target_language: Target language code (ISO 639-1)
            
        Returns:
            Translated text
        """
        # Convert language code to full name if needed
        target_lang_name = LanguageProcessor.LANGUAGE_MAP.get(
            target_language, 
            target_language
        )
        
        # For very long texts, split and translate in chunks
        if len(text) > 4000:
            chunks = LanguageProcessor._split_text(text)
            translated_chunks = []
            
            for chunk in chunks:
                # Translate each chunk
                translated_chunk = LanguageProcessor._translate_chunk(chunk, target_lang_name)
                translated_chunks.append(translated_chunk)
            
            return " ".join(translated_chunks)
        else:
            # Translate directly
            return LanguageProcessor._translate_chunk(text, target_lang_name)
    
    @staticmethod
    def _translate_chunk(text: str, target_language: str) -> str:
        """
        Translate a chunk of text using OpenAI
        
        Args:
            text: Text chunk to translate
            target_language: Target language name
            
        Returns:
            Translated text
        """
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"You are a professional translator. Translate the following text to {target_language}. Preserve formatting, line breaks, and special characters as much as possible. Translate only the content, not any metadata or markers."},
                    {"role": "user", "content": text}
                ],
                temperature=0.3,
                max_tokens=4000
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Translation error: {e}")
            return text  # Return original text if translation fails
    
    @staticmethod
    def _split_text(text: str, max_chunk_size: int = 4000) -> List[str]:
        """
        Split text into smaller chunks for processing
        
        Args:
            text: Text to split
            max_chunk_size: Maximum size of each chunk
            
        Returns:
            List of text chunks
        """
        # Try to split on paragraph breaks first
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            # If adding this paragraph would exceed chunk size, start a new chunk
            if len(current_chunk) + len(para) > max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If paragraph itself is too long, split by sentences
                if len(para) > max_chunk_size:
                    sentences = para.split('. ')
                    sentence_chunk = ""
                    
                    for sentence in sentences:
                        if len(sentence_chunk) + len(sentence) + 2 > max_chunk_size:
                            chunks.append(sentence_chunk)
                            sentence_chunk = sentence + ". "
                        else:
                            sentence_chunk += sentence + ". "
                    
                    if sentence_chunk:
                        current_chunk = sentence_chunk
                    else:
                        current_chunk = ""
                else:
                    current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        # Add the last chunk if there's anything left
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    @staticmethod
    def get_supported_languages() -> List[Dict[str, str]]:
        """
        Get list of supported languages
        
        Returns:
            List of language dictionaries with code and name
        """
        languages = []
        for code, name in LanguageProcessor.LANGUAGE_MAP.items():
            languages.append({
                "code": code,
                "name": name
            })
        
        # Sort by language name
        languages.sort(key=lambda x: x["name"])
        
        return languages
    
    @staticmethod
    def translate_transcript_segments(segments: List[Dict[str, Any]], target_language: str) -> List[Dict[str, Any]]:
        """
        Translate transcript segments to target language
        
        Args:
            segments: List of transcript segments with text
            target_language: Target language code
            
        Returns:
            List of translated transcript segments
        """
        translated_segments = []
        
        # Extract all texts to translate in batch
        all_texts = [segment["text"] for segment in segments]
        combined_text = "\n---SEGMENT BREAK---\n".join(all_texts)
        
        # Translate all at once
        translated_combined = LanguageProcessor.translate_text(combined_text, target_language)
        
        # Split back into segments
        translated_texts = translated_combined.split("\n---SEGMENT BREAK---\n")
        
        # Ensure we have the right number of translations
        if len(translated_texts) != len(segments):
            # Fallback: translate one by one
            for segment in segments:
                translated_text = LanguageProcessor.translate_text(segment["text"], target_language)
                translated_segment = segment.copy()
                translated_segment["text"] = translated_text
                translated_segments.append(translated_segment)
        else:
            # Use batch translations
            for i, segment in enumerate(segments):
                translated_segment = segment.copy()
                translated_segment["text"] = translated_texts[i]
                translated_segments.append(translated_segment)
        
        return translated_segments