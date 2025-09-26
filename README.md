# 🤖 Multilingual GemmaBot Chat

🔗 **Live App:** https://multilingualchatbotgroq.streamlit.app/

---

## ✨ Features

| Feature | Details |
|---------|---------|
| ⚡ **High-Speed Inference** | Utilizes Groq's **LPU™ Inference Engine** with the `Gemma2-9b-It` model for low-latency responses. |
| 🧠 **Conversational Memory** | Remembers names, topics, and context from previous messages within the session using LangChain’s in-memory store. |
| 🌐 **Multilingual Output** | Responds in the selected output language (Hindi, Japanese, German, etc.) regardless of input language. |
| ✂️ **Context Trimming** | Uses LangChain's `trim_messages` to prevent context overflow during long chats. |
| 🔄 **Session Control** | Start a **New Chat** with a button, or clear memory with the command `forget everything`. |
| 💬 **Real-time UX** | Shows *"GemmaBot is replying..."* while generating answers. |

---

## 🚀 Deployment and Access

| Link Type | URL |
|-----------|-----|
| **GitHub Repository** | [INSERT_YOUR_GITHUB_REPO_LINK_HERE] |
| **Live Streamlit App** | [INSERT_YOUR_DEPLOYED_STREAMLIT_URL_HERE] |

---

## ⚙️ Setup and Requirements

### 1. Prerequisites
- Get your **Groq API key** from [Groq Console](https://console.groq.com).
- Save it in an environment variable `GROQ_API_KEY`.

### 2. Project Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit app and LangChain orchestration logic. |
| `.env` | Stores your `GROQ_API_KEY`. |
| `requirements.txt` | Python dependencies. |

---

## 📦 requirements.txt
```txt
streamlit
python-dotenv
langchain-groq
langchain-core
langchain-community
```

---

## 🔑 .env File
```env
GROQ_API_KEY="your_groq_api_key_here"
```

---

## 👨‍💻 Installation and Usage

### Step 1: Clone the Repository
```bash
git clone [YOUR_REPO_URL]
```

### Step 2: Create Virtual Environment & Install Dependencies
```bash
python -m venv venv
source venv/bin/activate   # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Run the Streamlit App
```bash
streamlit run Chatbot.py
```

The app will open in your browser and is ready for use 🎉

---

## 💡 Code Explanation

| Component | Why It’s Used |
|-----------|---------------|
| **ChatGroq (Gemma2-9b-It)** | Fast inference LLM via Groq’s LPU engine. |
| **ChatMessageHistory** | In-memory session history to remember past messages. |
| **ChatPromptTemplate** | Defines assistant role + `{language}` variable for multilingual responses. |
| **trim_messages** | Trims conversation tokens (set to 200) to prevent context overflow. |
| **RunnableWithMessageHistory** | Wraps the chain so history is auto-fetched and updated. |
| **Streamlit** | Provides the UI (chat input, sidebar for language, buttons for new chat). |

---


---
