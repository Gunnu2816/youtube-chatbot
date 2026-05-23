import os
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
import gradio as gr

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def get_video_id(url):
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be" in url:
        return url.split("/")[-1].split("?")[0]
    return url

retriever = None
transcript_with_time = []
chat_history = []

def load_content(urls_text, manual_transcript):
    global retriever, transcript_with_time, chat_history
    transcript_with_time = []
    all_text = []

    if urls_text.strip():
        urls = [u.strip() for u in urls_text.strip().split("\n") if u.strip()]
        video_ids = [get_video_id(u) for u in urls]
        ytt_api = YouTubeTranscriptApi()
        for i, vid_id in enumerate(video_ids):
            try:
                tlist = ytt_api.fetch(vid_id)
                transcript_with_time += [{"text": t.text, "start": t.start, "video": i+1} for t in tlist]
                all_text.append(" ".join([t.text for t in tlist]))
            except:
                pass

    if not all_text:
        if manual_transcript.strip():
            all_text.append(manual_transcript.strip())
        else:
            return "❌ Auto fetch failed. Please paste transcript manually.", []

    transcript = " ".join(all_text)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = text_splitter.split_text(transcript)

    if not chunks:
        return "❌ Transcript is empty.", []

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    chat_history.clear()

    summary = ask_groq("Give me a short 3 line summary of this video")
    chat_history.clear()
    return f"✅ Loaded!\n\n📋 Summary:\n{summary}", []

def ask_groq(query):
    global chat_history
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs[:3]])
    timestamps = []
    for doc in docs[:3]:
        for t in transcript_with_time:
            if t["text"] in doc.page_content:
                minutes = int(t["start"] // 60)
                seconds = int(t["start"] % 60)
                timestamps.append(f"Video {t['video']} → {minutes}:{seconds:02d}")
                break
    chat_history.append({"role": "user", "content": query})
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based only on the provided YouTube video transcript context."},
            {"role": "user", "content": f"Transcript context:\n{context}"},
            *chat_history
        ],
        max_tokens=300
    )
    reply = response.choices[0].message.content
    chat_history.append({"role": "assistant", "content": reply})
    if timestamps:
        reply += f"\n\n⏱️ Relevant timestamps: {', '.join(set(timestamps))}"
    return reply

def chat(user_message, history):
    if not user_message.strip():
        return history, ""
    if retriever is None:
        history.append({"role": "assistant", "content": "Please load a video first!"})
        return history, ""
    reply = ask_groq(user_message)
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": reply})
    return history, ""

with gr.Blocks(title="🎬 YouTube Chatbot") as app:
    gr.Markdown("# 🎬 YouTube Chatbot\nChat with any YouTube video using AI!")
    with gr.Row():
        with gr.Column(scale=1):
            url_input = gr.Textbox(
                label="YouTube URLs (one per line)",
                placeholder="https://youtube.com/watch?v=...",
                lines=3
            )
            gr.Markdown("**OR paste transcript manually if auto-fetch fails:**")
            manual_input = gr.Textbox(
                label="📋 Manual Transcript",
                placeholder="Paste transcript here...",
                lines=6
            )
            load_btn = gr.Button("🚀 Load Video", variant="primary")
            status_box = gr.Textbox(label="Status", lines=5, interactive=False)
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat", height=450, type="messages")
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask anything about the video...",
                    label="Your Question",
                    scale=4
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

    gr.Markdown("""
    ### 📖 How to get transcript manually:
    1. Open the YouTube video
    2. Click **'...'** (3 dots) below the video
    3. Click **'Show transcript'**
    4. Click **3 dots inside transcript** → **Toggle timestamps OFF**
    5. Select all → Copy → Paste above
    """)

    load_btn.click(fn=load_content, inputs=[url_input, manual_input], outputs=[status_box, chatbot])
    send_btn.click(fn=chat, inputs=[msg_input, chatbot], outputs=[chatbot, msg_input])
    msg_input.submit(fn=chat, inputs=[msg_input, chatbot], outputs=[chatbot, msg_input])

app.launch()