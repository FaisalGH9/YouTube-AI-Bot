import streamlit as st
from src.langchain_pipeline.vector_store import create_vector_db_from_youtube_url
from src.langchain_pipeline.qa_chain import get_response_from_query, get_simple_response_from_query
from src.langchain_pipeline.summarizer import summarize_video_from_db
import textwrap
import time

st.set_page_config(page_title="🎬 YouTube AI Assistant", layout="centered")
st.title("🎬 YouTube AI Assistant")

with st.sidebar:
    with st.form(key='my_form'):
        youtube_url = st.text_area("🎥 YouTube URL", max_chars=200)
        query = st.text_area("❓ Ask something about the video", max_chars=200, key="query")

        audio_duration = st.selectbox(
            "⏱️ How much of the video to process?",
            ("Full video", "First 5 minutes", "First 10 minutes", "First 15 minutes", 
             "First 30 minutes", "First 60 minutes")  # Added more duration options
        )
        
        # Add processing quality option for long videos
        processing_quality = st.radio(
            "🔍 Processing Quality (for long videos)",
            ["Standard", "Fast (lower quality for 1hr+ videos)"]
        )

        mode = st.radio("🧠 Choose Mode", ["Question Answering", "Summarize Video"])
        submit_button = st.form_submit_button(label="🚀 Submit")

if submit_button and youtube_url:
    if not youtube_url.startswith("https://www.youtube.com") and not youtube_url.startswith("https://youtu.be"):
        st.warning("⚠️ Please enter a valid YouTube URL.")
    else:
        try:
            # Create progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Define callback function to update progress
            def update_progress(step, total_steps, status, details=None):
                progress_value = step / total_steps
                progress_bar.progress(progress_value)
                status_message = f"⏳ {status}... ({step}/{total_steps})"
                if details:
                    status_message += f" - {details}"
                status_text.text(status_message)
                
            with st.spinner("🔊 Processing video..."):
                # Simulate progress for standard approach
                update_progress(1, 4, "Downloading audio")
                time.sleep(0.5)
                update_progress(2, 4, "Processing audio")
                
                # Use quality setting to determine bitrate
                bitrate = "6k" if processing_quality == "Fast (lower quality for 1hr+ videos)" else "8k"
                
                db, audio_size = create_vector_db_from_youtube_url(
                    youtube_url,
                    duration_choice=audio_duration,
                )
                    
                update_progress(4, 4, "Completed processing")
                
                if audio_size > 0:
                    st.success(f"🎧 New audio processed! Compressed size: {audio_size:.2f} MB")
                else:
                    st.info("📦 Using previously processed transcript (from ChromaDB cache)")

            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()

            if mode == "Summarize Video":
                with st.spinner("📝 Summarizing video content..."):
                    # Create expandable section for advanced options
                    with st.expander("Advanced Summarization Options"):
                        summary_model = st.selectbox(
                            "Select model for summarization",
                            ["gpt-3.5-turbo-instruct", "gpt-4o"],
                            index=0
                        )
                        
                        summary_length = st.select_slider(
                            "Summary length",
                            options=["Brief", "Moderate", "Detailed"],
                            value="Moderate"
                        )
                    
                    # Adjust parameters based on user choices (if they expanded the advanced options)
                    selected_model = summary_model if 'summary_model' in locals() else "gpt-3.5-turbo-instruct"
                    selected_length = summary_length if 'summary_length' in locals() else "Moderate"
                    
                    try:
                        response = summarize_video_from_db(db, model_name=selected_model, summary_length=selected_length)
                        st.subheader("📋 Summary:")
                        st.text(textwrap.fill(response, width=85))
                    except Exception as e:
                        st.error(f"❌ Error during summarization: {str(e)}")
                        st.info("Trying alternative summarization approach for very long videos...")
                        
                        # Fallback to a more conservative approach
                        try:
                            response = summarize_video_from_db(db, model_name="gpt-3.5-turbo-instruct", summary_length="Brief")
                            st.subheader("📋 Summary (Reduced):")
                            st.text(textwrap.fill(response, width=85))
                        except Exception as e2:
                            st.error("Unable to generate summary. The video may be too long or complex.")
                            st.info("Try using the Question Answering mode instead, which can handle longer content better.")
            else:
                if query.strip() == "":
                    st.warning("⚠️ Please enter a question.")
                    st.stop()

                with st.spinner("💬 Thinking..."):
                    # Create expandable section for advanced options
                    with st.expander("Advanced Q&A Options"):
                        qa_model = st.selectbox(
                            "Select model for Q&A",
                            ["gpt-3.5-turbo-instruct", "gpt-4o"],
                            index=0
                        )
                        
                        context_size = st.slider(
                            "Context size (chunks)",
                            min_value=1,
                            max_value=5,
                            value=2
                        )
                    
                    # Use selected model if available
                    selected_model = qa_model if 'qa_model' in locals() else "gpt-3.5-turbo-instruct"
                    selected_k = context_size if 'context_size' in locals() else 2
                    
                    # First try the regular approach with token management
                    try:
                        response = get_response_from_query(db, query, k=selected_k, model_name=selected_model)
                        st.subheader("💡 Answer:")
                        st.text(textwrap.fill(response, width=85))
                        
                        st.markdown("### 📄 Matched Transcript Snippets:")
                        for doc in db.similarity_search(query, k=min(2, selected_k)):
                            # Limit the display to prevent UI overload
                            snippet = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                            st.code(textwrap.fill(snippet, width=80))
                            
                    except Exception as e:
                        st.error(f"❌ Error during Q&A: {str(e)}")
                        st.info("Try reducing the context size or using a different model.")
                        
                        # Automatically fall back to simplified approach
                        try:
                            response = get_simple_response_from_query(db, query, model_name="gpt-3.5-turbo-instruct")
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