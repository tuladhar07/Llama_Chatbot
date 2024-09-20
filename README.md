# PalmMindbot - Conversational PDF Assistant

Welcome to **PalmMindbot**, a conversational AI application designed to assist with PDF-based queries and appointment scheduling. This project allows users to upload PDF documents, ask questions related to the content, and even book appointments through a user-friendly interface powered by Streamlit. The system also provides conversational features, where it can store conversation history and interact dynamically with users.

## Why LLaMA Instead of OpenAI?
I chose to use the LLaMA model in this project instead of OpenAI's API due to the cost implications of using OpenAI. OpenAI's API incurs charges based on usage, which can accumulate over time, especially for a conversational assistant that processes many queries. By opting for LLaMA, a local model, I can reduce the cost associated with running the bot while maintaining strong performance for natural language understanding and generation tasks. This decision ensures the bot remains cost-effective for long-term use.

## Features

1. **PDF Upload**: Users can upload a PDF file, and the bot processes it to extract and split the content into manageable text chunks.
2. **Conversational Interface**: The bot engages with the user in a conversational format, displaying previous conversations and handling both document-based queries and appointment bookings.
3. **Question-Answering**: Users can ask questions related to the content of the uploaded PDF, and the bot provides answers by retrieving relevant information using sentence embeddings and a FAISS vector store.
4. **Appointment Booking**: The bot can handle appointment booking by gathering user information such as name, email, phone number, and preferred appointment date.
5. **Streamlit UI**: The bot is hosted in a Streamlit web application with a responsive user interface, featuring real-time conversation updates, submission forms, and an intuitive sidebar for PDF upload.
6. **Embedding and Retrieval**: The project uses the `SentenceTransformer` model for generating embeddings from PDF text chunks, and FAISS for efficient similarity search.

## Tech Stack

- **Python**: Main programming language used for building the bot.
- **Streamlit**: Framework for creating the web interface.
- **LangChain**: Utilized for text splitting and managing larger document queries.
- **Ollama**: A backend model for generating responses based on context.
- **FAISS**: An index for efficient nearest-neighbor search to find relevant text chunks.
- **PyPDF2**: Used for reading and extracting text from PDFs.
- **Dateparser**: For handling natural language date inputs.
- **Sentence Transformers**: Used to generate text embeddings.

## Getting Started

### Prerequisites

Make sure you have the following installed on your system:

- Python 3.8+
- Streamlit
- PyPDF2
- SentenceTransformers
- FAISS
- Ollama (Ensure it's properly installed and configured)
- Dateparser
- Numpy

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tuladhar07/PalmMind_Chatbot/palmmindbot.git
   
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

4. Run the application:
   ```bash
   streamlit run main.py
