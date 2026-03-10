# Smart_Speak
Smart Speak is an innovative application designed to analyze and evaluate speech recordings using AI-powered RAG (Retrieval Augmented Generation) technology. It provides users with comprehensive insights into their speaking performance by comparing their speech against reference documents. The application analyzes aspects such as content accuracy, completeness, clarity, and provides specific line-by-line improvement suggestions.

## Features

🎯 **RAG-Based Evaluation**: Upload reference documents (PDF/TXT) and evaluate your speech against them using FAISS vector store and semantic search

📊 **Comprehensive Analysis**: Get detailed feedback including:
- Overall score and performance summary
- Strengths and weaknesses identification
- Specific line-by-line improvement suggestions
- Common mistakes highlighted
- Chunk-by-chunk detailed analysis with accuracy, completeness, and clarity scores

🎙️ **Advanced Speech Processing**: Uses OpenAI Whisper for accurate speech-to-text transcription

🤖 **AI-Powered Feedback**: Leverages Google Gemini 2.0 Flash for intelligent evaluation and actionable suggestions


## Instructions to Run the App
1. **Clone the Repository**: Start by cloning the Smart Speak repository to your local machine.

   ```bash
   git clone https://github.com/NinadGawali/Smart_Speak
   ```
2. **Navigate to the Project Directory**: Change your current directory to the Smart Speak project folder.

   ```bash
    cd Smart_Speak
    ```
4. **Set Up Environment Variables**: Create a `.env` file in the root directory of the project and add your API keys or any necessary environment variables. For example:
    ```
    GEMINI_API_KEY = "your_api_key_here"
    ```
5. **Create a Virtual Environment**: It is recommended to create a virtual environment to manage dependencies.

   ```bash
   python -m venv venv
   venv/bin/activate  
   ```

3. **Install Dependencies**: Install the required dependencies using pip.
    ```bash
    pip install -r requirements.txt
    ```
6. **Run the Application**: Start the Streamlit application.
    ```bash
    streamlit run llm_inference/speech_info_analysis.py
    ```
7. **Access the App**: Open your web browser and navigate to `http://localhost:8501` to access the Smart Speak application.

## Notes
- Ensure you have the necessary API keys (GEMINI_API_KEY) in your `.env` file.
- The application needs some time to load at first, especially when:
  - Processing PDF/TXT documents
  - Creating vector embeddings (first-time document upload)
  - Loading Whisper model for transcription
- Please be patient while it initializes.

## Usage Flow
1. **Upload Reference Document**: Upload a PDF or TXT file containing the topic/speech content
2. **Wait for Processing**: The app will create a knowledge base from your document
3. **Record Your Speech**: Use the audio recorder to capture your speech
4. **Analyze**: Click "Transcribe & Analyze with RAG" to get comprehensive feedback
5. **Review Results**: 
   - View overall score and summary
   - Check strengths and weaknesses
   - Review specific line-by-line improvement suggestions
   - Identify common mistakes
   - Explore detailed chunk-by-chunk analysis

## Technical Stack
- **Speech-to-Text**: OpenAI Whisper (base model)
- **Embeddings**: Google Generative AI Embeddings
- **Vector Store**: FAISS (CPU version)
- **LLM**: Google Gemini 2.0 Flash
- **Framework**: Streamlit
- **Document Processing**: PyPDF, LangChain