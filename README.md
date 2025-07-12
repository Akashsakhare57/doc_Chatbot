
# 🤖 Harry: Groq-Powered Document Intelligence Chatbot

An intelligent, interactive chatbot powered by **Groq + LangChain**, designed to understand and summarize long-form PDF documents (100+ pages) and hold multi-turn conversations using a friendly **IPython widgets interface**.

## 🎯 Overview

This project combines lightning-fast Groq inference with advanced **Retrieval-Augmented Generation (RAG)** techniques and a **conversational GUI** built using `ipywidgets` in Jupyter or Google Colab.

## ✨ Key Features

- 📄 **PDF Upload & Chunking**
- 🔍 **Hybrid Retrieval** (FAISS + BM25)
- 💬 **Conversational Memory** support
- 🎨 **ELI5 Mode** – Explain like I’m 5
- 🧠 **Show History** & **Clear Chat**
- ⌨️ **Submit on Enter** or via buttons

---

## 🏗️ Architecture

```text
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Document      │    │   Processing    │    │   Vector Store  │
│   Ingestion     │───▶│   Pipeline      │───▶│   (FAISS)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │───▶│   Hybrid RAG    │───▶│   Groq AI       │
│   Interface     │    │   Retrieval     │    │   (Llama 3.1)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Response      │
                       │   Generation    │
                       └─────────────────┘
```
```

---

## 🛠️ Technical Stack

- **LLM**: Groq (Llama 3.1 70B)
- **Framework**: LangChain
- **Embeddings**: HuggingFace/Nomic
- **Storage**: FAISS, BM25, Chroma optional
- **Frontend**: IPython Widgets
- **Memory**: `ConversationBufferMemory`

---

## 📦 Tools

- Langchain
- Chroma DB
- Groq API
- Hugging Face

---

## 🚀 Getting Started

You can run this in **Jupyter Notebook** or **Google Colab**.

### 🧠 Launch the Interface

You’ll get a clean UI with:

- A question input field
- Buttons: `Ask Harry`, `ELI5 Mode`, `Clear Chat`, `Show History`
- Real-time output area

---

## 💡 Usage Examples

| Action          | What to Do                                  |
|-----------------|----------------------------------------------|
| Ask a question  | Type and press `Enter` or click `Ask Harry` |
| ELI5 Mode       | Click `ELI5 Mode` for simplified explanation |
| Clear Chat      | Clears conversation memory                  |
| Show History    | Prints previous questions + answers         |
| Upload Document | Handled in separate interface (optional)    |

---

## 📁 File Structure

```text
genai-chatbot/
├── README.md
├── requirements.txt
├── Gen_AI_Chatbot_clean.py
├── chroma
├── data
│   ├── Build_Don't_Talk_.pdf
```

---

## 🔧 To-Do / Future Enhancements

- [ ] Add multi-document support
- [ ] Add download transcript/history
- [ ] Add summarization for full documents
- [ ] Integrate visual response for charts/tables
- [ ] Support streamlit or gradio interface

---

> *Built with ❤️ using Groq AI and LangChain*
