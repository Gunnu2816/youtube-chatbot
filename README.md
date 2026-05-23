# YouTube Chatbot

This project lets users chat with YouTube videos using AI.

It extracts transcripts from YouTube videos, converts them into embeddings using LangChain + FAISS, and answers questions using Groq LLMs.

I built this project to learn:
- Retrieval-Augmented Generation (RAG)
- Vector databases
- LLM API integration
- Semantic search
- AI application deployment

## Features

- Ask questions from YouTube videos
- Automatic transcript extraction
- Manual transcript input if auto-fetch fails
- Timestamp-based responses
- Multiple video support
- Gradio-based UI

## Tech Used

- Python
- Gradio
- LangChain
- FAISS
- HuggingFace Embeddings
- Groq API

## Run Locally

```bash
pip install -r requirements.txt
python app.py
```

## Live Demo

https://gungun2826-youtube-chatbot.hf.space
