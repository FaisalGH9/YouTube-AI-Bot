import streamlit as st
import textwrap
import time
from src.langchain_pipeline.processor import VideoProcessor
from src.ui.progress import ProgressManager
from src.config.settings import MAX_CONCURRENT_REQUESTS

# Set page configuration
st.set_page_config(page_title="🎬 YouTube AI Assistant", layout="centered")
st.title("🎬 YouTube AI Assistant")

# Initialize components
processor = VideoProcessor()
progress_mgr = ProgressManager()

with st.sidebar:
    with st.form(key='my_form'):
        youtube_url = st.text_area("🎥 YouTube URL", max_chars=200)
        query = st.text_area("❓ Ask something about the video", max_chars=200, key="query")

        # Duration options
        audio_duration = st.selectbox(
            "⏱️ How much of the video to process?",
            ("Full video", "First 5 minutes", "First 10 minutes", "First 15 minutes", 
             "First 30 minutes", "First 60 minutes")
        )
        
        # Processing quality option
        processing_quality = st.radio(
            "🔍 Processing Quality (for long videos)",
            ["Standard", "Fast (lower quality for 1hr+ videos)"]
        )
        
        # Add parallel processing option
        parallelization = st.slider(
            "🚀 Parallel Processing (faster, uses more API calls)",
            min_value=1,
            max_value=5,
            value=3,
            help="Higher values process faster but consume more API credits"
        )

        # Mode selection
        mode = st.radio("🧠 Choose Mode", ["Question Answering", "Summarize Video"])
        
        # Submit button
        submit_button = st.form_submit_button(label="🚀 Submit")

if submit_button and youtube_url:
    # Validate YouTube URL
    if not youtube_url.startswith("https://www.youtube.com") and not youtube_url.startswith("https://youtu.be"):
        st.warning("⚠️ Please enter a valid YouTube URL.")
    else:
        try:
            # Initialize progress display
            progress_display = progress_mgr.initialize()
            
            # Set parallelization level in processor
            processor.set_parallelization(parallelization)
            
            # Process video
            db, audio_size = processor.process_video(
                youtube_url,
                duration_choice=audio_duration,
                progress_callback=progress_display.update
            )
            
            # Clear progress display
            progress_display.clear()
            
            # Display success message
            if audio_size > 0:
                st.success(f"🎧 New audio processed! Compressed size: {audio_size:.2f} MB")
            else:
                st.info("📦 Using previously processed transcript (from ChromaDB cache)")

            # Process request based on selected mode
            if mode == "Summarize Video":
                with st.spinner("📝 Summarizing video content..."):
                    try:
                        # Use default moderate summary length
                        response = processor.summarize_video(
                            db, 
                            model_name="gpt-3.5-turbo-instruct", 
                            summary_length="Moderate"
                        )
                        
                        # Display summary
                        st.subheader("📋 Summary:")
                        st.text(textwrap.fill(response, width=85))
                        
                    except Exception as e:
                        # Handle summarization errors
                        st.error(f"❌ Error during summarization: {str(e)}")
                        st.info("Trying alternative summarization approach for very long videos...")
                        
                        # Fallback to brief summary with simpler model
                        try:
                            response = processor.summarize_video(
                                db, 
                                model_name="gpt-3.5-turbo-instruct", 
                                summary_length="Brief"
                            )
                            st.subheader("📋 Summary (Reduced):")
                            st.text(textwrap.fill(response, width=85))
                            
                        except Exception as e2:
                            st.error("Unable to generate summary. The video may be too long or complex.")
                            st.info("Try using the Question Answering mode instead, which can handle longer content better.")
                            
            else:  # Question Answering mode
                # Validate query
                if query.strip() == "":
                    st.warning("⚠️ Please enter a question.")
                    st.stop()

                with st.spinner("💬 Thinking..."):
                    try:
                        # Use default settings
                        response, relevant_docs = processor.answer_question(
                            db, 
                            query, 
                            k=3,  # Use a moderate context size
                            model_name="gpt-3.5-turbo-instruct" 
                        )
                        
                        # Display answer
                        st.subheader("💡 Answer:")
                        st.text(textwrap.fill(response, width=85))
                        
                        # Show relevant transcript snippets
                        st.markdown("### 📄 Matched Transcript Snippets:")
                        for doc in relevant_docs[:2]:  # Limit to 2 snippets for clarity
                            # Limit display length
                            snippet = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                            st.code(textwrap.fill(snippet, width=80))
                            
                    except Exception as e:
                        # Handle QA errors
                        st.error(f"❌ Error during Q&A: {str(e)}")
                        
                        # Automatically fall back to simplified approach
                        time.sleep(1)  # Brief pause to show the error message
                        with st.spinner("Trying simplified approach..."):
                            try:
                                response, _ = processor.answer_question(
                                    db, query, k=1, model_name="gpt-3.5-turbo-instruct"
                                )
                                st.subheader("💡 Answer (Simplified):")
                                st.text(textwrap.fill(response, width=85))
                                
                            except Exception as e2:
                                st.error("Unable to process this question with the current video.")
                                st.info("Try asking a more specific question or processing a shorter segment of the video.")

        except FileNotFoundError as e:
            st.error(str(e))
        except RuntimeError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            st.info("If processing a very long video, try the 'Fast' processing option or select a shorter duration.")

# Footer
st.markdown("---")
st.markdown("#### YouTube AI Assistant")
st.markdown("This tool helps you extract information from YouTube videos through transcription and AI processing.")