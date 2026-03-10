# Smart Speak - RAG-Based Speech Evaluator Usage Guide

## Overview
The Smart Speak application now uses **RAG (Retrieval Augmented Generation)** technology to provide comprehensive speech evaluation. Instead of typing expected content, you upload reference documents and the system creates a knowledge base to evaluate your speech.

## How It Works

### 1. Document Upload & Knowledge Base Creation
- **Supported Formats**: PDF and TXT files
- **Process**:
  - Upload your reference document (e.g., lecture notes, speech script, topic material)
  - The system splits the document into chunks (500 characters with 100 character overlap)
  - Each chunk is converted into vector embeddings
  - All embeddings are stored in a FAISS vector store for fast similarity search

### 2. Speech Recording
- Use the built-in audio recorder to capture your speech
- Audio is saved as WAV format in the `recordings/` folder
- You can review the recording before analysis

### 3. Speech-to-Text Transcription
- The application uses **OpenAI Whisper (base model)** for accurate transcription
- Supports various audio qualities and speaking styles
- Handles mono and stereo audio automatically

### 4. Chunked Analysis with RAG
- **Transcript Chunking**: Your speech is split into smaller segments (3 sentences per chunk)
- **Semantic Search**: For each chunk, the system retrieves the 3 most relevant sections from the reference document
- **Evaluation**: Each chunk is evaluated against the retrieved context for:
  - **Accuracy** (0-100): How well does it match the reference content?
  - **Completeness** (0-100): Are all key points covered?
  - **Clarity** (0-100): Is it well articulated?

### 5. Comprehensive Feedback Report

The evaluation provides multiple layers of feedback:

#### A. Overall Score (0-100)
- Aggregated performance across all chunks
- Color-coded: Green (80+), Yellow (60-79), Red (<60)

#### B. Summary
- High-level assessment of your speech performance
- Key takeaways about your delivery

#### C. Strengths
- What you did well
- Aspects that matched the reference material effectively

#### D. Areas for Improvement
- Specific weaknesses identified
- Content gaps or inaccuracies

#### E. Specific Line-by-Line Improvements
- **Most Valuable Feature**: Identifies exact lines from your transcript that need improvement
- Provides specific suggestions on how to rephrase or enhance each line
- Format:
  ```
  Original Line: "Your actual spoken words..."
  Suggestion: "Try saying this instead because..."
  ```

#### F. Common Mistakes
- Patterns of errors identified across your speech
- Examples: "Repeated use of filler words", "Missed key terminology", etc.

#### G. Chunk-by-Chunk Detailed Analysis
- Expandable sections for each speech segment
- Shows:
  - The exact text you spoke
  - Individual scores for accuracy, completeness, clarity
  - Specific feedback for that chunk
  - Issues identified in that segment

## Example Use Cases

### 1. Academic Presentations
- Upload your lecture slides or notes as PDF
- Record your practice presentation
- Get feedback on technical accuracy and completeness

### 2. Public Speaking Practice
- Upload your speech script as TXT
- Record your delivery
- Identify which parts need improvement

### 3. Training & Coaching
- Upload training material or curriculum
- Record training sessions
- Verify content coverage and accuracy

### 4. Interview Preparation
- Upload job description or topic notes
- Practice your responses
- Get feedback on relevance and clarity

## Tips for Best Results

1. **Quality Reference Documents**:
   - Use clear, well-formatted PDFs
   - Ensure TXT files are properly encoded (UTF-8)
   - Include comprehensive coverage of the topic

2. **Recording Quality**:
   - Use a quiet environment
   - Speak clearly and at moderate pace
   - Ensure microphone is properly positioned

3. **Speech Structure**:
   - Follow a logical flow matching your reference document
   - Use clear transitions between topics
   - Speak in complete sentences

4. **Interpreting Results**:
   - Focus on specific improvement suggestions
   - Review low-scoring chunks for targeted practice
   - Use common mistakes section to identify patterns

## Technical Details

### Vector Store
- **Type**: FAISS (Facebook AI Similarity Search)
- **Storage**: Local filesystem (`llm_inference/vectorstore/`)
- **Embeddings**: Google Generative AI Embeddings (model: embedding-001)
- **Purpose**: Fast semantic similarity search for chunk-to-document matching

### Evaluation Model
- **LLM**: Google Gemini 2.0 Flash (Experimental)
- **Temperature**: 0 (deterministic, consistent results)
- **Output**: Structured JSON with detailed metrics

### Processing Pipeline
```
Document → Chunks → Embeddings → Vector Store
                                      ↓
Audio → Whisper → Transcript → Chunks
                                      ↓
                              RAG Retrieval
                                      ↓
                              Gemini Evaluation
                                      ↓
                            Comprehensive Report
```

## Troubleshooting

### Document Processing Fails
- Check file format (PDF/TXT only)
- Verify file is not corrupted
- Ensure sufficient disk space

### Transcription Errors
- Check audio quality and volume
- Speak more clearly
- Reduce background noise

### Low Evaluation Scores
- Ensure your speech covers the reference material
- Speak more clearly and completely
- Use terminology from the reference document
- Follow the structure of the reference material

### Vector Store Issues
- Check `llm_inference/vectorstore/` directory permissions
- Ensure GEMINI_API_KEY is valid
- Verify internet connection for embedding generation

## Future Enhancements

Potential features for future versions:
- Multi-language support
- Real-time evaluation during recording
- Custom evaluation criteria
- Export reports as PDF
- Historical performance tracking
- Voice quality analysis (pace, tone, pauses)
- Comparison with multiple reference documents

---

For issues or feature requests, please check the GitHub repository.
