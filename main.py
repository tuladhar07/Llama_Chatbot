import streamlit as st
import pickle
import re
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama  # Ensure ollama is installed and configured
import os
import faiss
import dateparser  # For parsing dates
import datetime

# Sidebar Contents
with st.sidebar:
    # Add the logo next to the title
    st.markdown(
        """
        <div style="display: flex; align-items: center;">
            <img src="https://palmmind.com/wp-content/uploads/2022/01/Palm-Mind-Technology-01.png" style="width: 100px; height: 70px; margin-right: 0px;">
            <h1 style="margin: 0;">bot</h1>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Display a subtitle in smaller font
    st.markdown('<h5 style="margin: 0;">This is a Conversational Docs Bot</h5>', unsafe_allow_html=True)
    
    add_vertical_space(1)  # Add some vertical space
    pdf = st.file_uploader("Upload the PDF", type='pdf')  # Allow user to upload a PDF file
    add_vertical_space(2)
    
    # Author info aligned to the left in smaller font
    st.markdown('<h6 style="text-align: left; margin: 0;">Made by Nhujaw Tuladhar</h6>', unsafe_allow_html=True)

# The main function
def main():
    # Initialize session state variables if they don't already exist
    if 'conversation' not in st.session_state:
        st.session_state['conversation'] = []  # Store conversation history
    if 'vectorstore' not in st.session_state:
        st.session_state['vectorstore'] = None  # Placeholder for the vectorstore
    if 'chunks' not in st.session_state:
        st.session_state['chunks'] = None  # To hold text chunks from the PDF
    if 'user_info' not in st.session_state:
        st.session_state['user_info'] = {
            "name": None,
            "email": None,
            "phone": None,
            "appointment_date": None
        }  # Store user information for appointment booking
    if 'current_question' not in st.session_state:
        st.session_state['current_question'] = None  # Track the current question for user info collection

    # Step 1: Display welcome message and instructions
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Welcome to PalmMindbot!</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #555;'>🤖 Ask any question about the PDF or book an appointment 📅</h2>", unsafe_allow_html=True)

    # Display previous queries and responses
    # Display previous queries and responses
    st.subheader("Conversation")
    for query, resp in st.session_state['conversation']:
        # Create a container for each message
        st.markdown("""
            <div style="margin-bottom: 10px;">
                <div style='background-color: #d1e7dd; color: #0f5132; border-radius: 8px; padding: 10px; text-align: right;'>
                    <strong>You:</strong> {query}
                </div>
                <div style='background-color: #e9ecef; border-radius: 8px; padding: 10px; margin-bottom: 5px;'>
                    <strong>Assistant:</strong> {resp}
                </div>
            </div>
        """.format(query=query, resp=resp), unsafe_allow_html=True)


    # New query input box at the bottom
    with st.form(key='query_form', clear_on_submit=True):
        # Adding custom CSS to style the input and button
        st.markdown("""
            <style>
            .stTextInput > div > input {
                border: 1px solid #d1d1d1;
                border-radius: 5px;  # Rounded corners
                padding: 12px 15px;  # Padding for input field
                font-size: 16px;  # Font size
                width: calc(100% - 30px);  # Full width minus padding
                margin-right: 10px;  # Space between input and button
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);  # Subtle shadow
                transition: border 0.2s ease-in-out;  # Transition effect
            }
            .stTextInput > div > input:focus {
                outline: none;  # Remove outline on focus
                border: 1px solid #4CAF50;  # Change border color on focus
            }
            .stButton > button {
                background-color: #4CAF50;  # Button background color
                color: white;  # Button text color
                border: none;  # No border
                border-radius: 5px;  # Rounded corners
                padding: 12px 20px;  # Padding for button
                font-size: 16px;  # Font size
                cursor: pointer;  # Pointer cursor on hover
                transition: background-color 0.2s ease-in-out;  # Transition effect
            }
            .stButton > button:hover {
                background-color: #45a049;  # Darker green on hover
            }
            </style>
        """, unsafe_allow_html=True)

        # Input field for user query
        user_query = st.text_input(
            "Enter your query:",
            key='user_query_input',
            placeholder="Type your question here..."  # Placeholder text
        )
        
        # Submit button
        submit_button = st.form_submit_button(
            label="Submit",
            help="Click to send your query."  # Tooltip for button
        )

        # Check if the submit button is clicked and user_query is not empty
        if submit_button and user_query:
            # Check if the user is providing their information
            if st.session_state['current_question'] is not None:
                if st.session_state['current_question'] == "name":
                    st.session_state['conversation'].append((user_query, f"Got it, {user_query}! Now, could you please provide your email address?"))
                    st.session_state['user_info']['name'] = user_query  # Store name
                    st.session_state['current_question'] = "email"  # Move to next question
                    #st.session_state['conversation'].append((user_query, f"Got it, {user_query}! Now, could you please provide your email address?"))
                elif st.session_state['current_question'] == "email":
                    # Validate email format
                    if re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", user_query):
                        st.session_state['user_info']['email'] = user_query  # Store email
                        st.session_state['current_question'] = "phone"  # Move to next question
                        st.session_state['conversation'].append((user_query, "Thank you! Finally, could you provide your phone number?"))
                    else:
                        st.session_state['conversation'].append((user_query, "Please enter a valid email address."))
                elif st.session_state['current_question'] == "phone":
                    # Validate phone number format
                    if re.match(r"^\d{10,15}$", user_query):
                        st.session_state['user_info']['phone'] = user_query  # Store phone number
                        st.session_state['current_question'] = "appointment_date"  # Move to next question
                        st.session_state['conversation'].append((user_query, "Thank you! When would you like to book the appointment? Please provide a date."))
                    else:
                        st.session_state['conversation'].append((user_query, "Please enter a valid phone number."))
                elif st.session_state['current_question'] == "appointment_date":
                    # Parse and validate date
                    parsed_date = dateparser.parse(user_query, settings={'PREFER_DATES_FROM': 'future',  'RETURN_AS_TIMEZONE_AWARE': False})
                    if parsed_date is not None:
                        now = datetime.datetime.now()
                        if parsed_date.date() >= now.date():  # Check if date is in the future
                            st.session_state['user_info']['appointment_date'] = parsed_date.strftime("%Y-%m-%d")
                            st.session_state['current_question'] = None  # Reset for future use
                            st.session_state['conversation'].append((user_query, f"<span style='color: green;'>Thank you! Your appointment has been booked for {st.session_state['user_info']['appointment_date']}.</span>"))
                        else:
                            st.session_state['conversation'].append((user_query, "Please provide a date that is in the future."))
                    else:
                        st.session_state['conversation'].append((user_query, "Please provide a valid date."))

            else:
                # Handle regular queries if not collecting user info
                if "call me" in user_query.lower():  # Check if "call me" is present in the input
                    st.session_state['current_question'] = "name"  # Start collecting user info
                    st.session_state['conversation'].append((user_query, "Great! Let's get your details. What's your name?"))

                else:
                    # Handle PDF processing if available
                    if pdf is not None and st.session_state['chunks'] is None:
                        pdf_reader = PdfReader(pdf)  # Read the uploaded PDF

                        text = ""
                        for page in pdf_reader.pages:
                            page_text = page.extract_text()  # Extract text from each page
                            if page_text:
                                text += page_text

                        # Split text into chunks for processing
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=200,
                            length_function=len
                        )
                        st.session_state['chunks'] = text_splitter.split_text(text=text)

                        # Generate embeddings for the chunks
                        model = SentenceTransformer('all-MiniLM-L6-v2')
                        embeddings = model.encode(st.session_state['chunks'])
                        embeddings = np.array(embeddings).astype('float32')
                        vectorstore = faiss.IndexFlatL2(embeddings.shape[1])  # Create a FAISS index
                        vectorstore.add(embeddings)  # Add embeddings to the index

                        store_name = pdf.name[:4]  # Get a short name for the file
                        if os.path.exists(f"{store_name}.pkl"):  # Check if embeddings are already saved
                            with open(f"{store_name}.pkl", "rb") as f:
                                st.session_state['vectorstore'] = pickle.load(f)  # Load existing embeddings
                            st.write('Embedding loaded from disk.')
                        else:
                            with open(f"{store_name}.pkl", "wb") as f:
                                pickle.dump(vectorstore, f)  # Save new embeddings
                            st.session_state['vectorstore'] = vectorstore
                            st.write('Embedding computation completed.')

                    # Generate embedding for the user query
                    query_embedding = SentenceTransformer('all-MiniLM-L6-v2').encode([user_query])
                    query_embedding = np.array(query_embedding).astype('float32')

                    # Perform search if vectorstore is available
                    if st.session_state['vectorstore'] is not None:
                        distances, indices = st.session_state['vectorstore'].search(query_embedding, k=5)  # Search for relevant chunks

                        if indices.size > 0:
                            retrieved_docs = [st.session_state['chunks'][i] for i in indices[0] if i < len(st.session_state['chunks'])]  # Get relevant documents
                            context = " ".join(retrieved_docs)  # Combine them into context
                        else:
                            context = "No relevant documents found."  # Fallback message
                    else:
                        context = "No PDF uploaded. Answering with general knowledge."  # Fallback message

                    # Interact with the Ollama model to generate a response
                    client = ollama.Client()
                    response = client.chat(
                        model="llama2",
                        messages=[  # Prepare message context for the model
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": f"Context: {context}\n\nQuestion: {user_query}"}
                        ]
                    )

                    response_content = response.get('message', {}).get('content', 'No content available')  # Extract the response content

                    # Display the assistant's response
                    st.write(f"**Assistant:** {response_content}")

                    # Update conversation history with the new exchange
                    st.session_state['conversation'].append((user_query, response_content))

            # Limit conversation history to the last 10 exchanges to keep UI clean
            st.session_state['conversation'] = st.session_state['conversation'][-10:]
            st.rerun()  # Refresh the page to display updated content

if __name__ == '__main__':
    main()  
