import json
import os
import requests
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core import SimpleDirectoryReader
import google.generativeai as genai
from copy import deepcopy
from tempfile import NamedTemporaryFile
import tempfile

def create_index(post_url, project_id, env_id, input_text):

    headers = {'projectId': project_id, 'environmentId': env_id}
    json_data = {"input_sentence": input_text}
    response = requests.post(post_url, headers=headers, json=json_data)
    return response.json()


def generate_index_entry(index_creation_url, projectId, environmentId, documents):
    # st.write(documents)
    for doc in documents:
        if doc.page_content:
            create_index(index_creation_url, 
                         projectId, 
                         environmentId, 
                         str(doc.page_content))
    return 1


# Function to make the API request
def search_pdf_data(search_url, project_id, env_id, query, limit=5, offset=0):

    # Headers with projectId and environmentId
    headers = {
        'projectId': project_id,
        'environmentId': env_id
    }
    
    # Parameters for the search
    json_data = {
        "query": query,
        "limit": limit,
        "offset": offset
    }
    
    # Make the API request
    response = requests.get(search_url, headers=headers, params=json_data)
    
    # Check if the request was successful
    if response.status_code == 200:
        result_data = response.json()
        # Extract context from response
        context = "\n".join(item['input_sentence'] for item in result_data[0]['data'])
        return context
    else:
        print('Failed:', response.status_code, response.text)
        return None


def generate_ai_response(context, user_query):
    # System and user prompts
    system_prompt = """You are a helpful assistant who is expert at answering user's queries based on the context"""
    
    user_prompt = f"""
    Provide a relevant, informative response to the user's query using the given context.
    - Answer directly without referring the user to any external links.
    - Use an unbiased, journalistic tone and avoid repeating text.
    - Format your response in markdown with bullet points for clarity.
    - If the exact result is not available based on the provided context return the closest matching result.
    - If no relevant result is found, inform the user that there are no results for their query.

    Context Block:
    {context}

    User Query:
    {user_query}
    """
    
    # Configure GenAI model and generate response
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "response_mime_type": "text/plain",
    }

    script_model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        generation_config=generation_config,
        system_instruction=system_prompt
    )
    
    response = script_model.generate_content(user_prompt)
    
    if response:
        return response.text
    else:
        print("Request failed")
        return '0'


def main():

    index_placeholder = None
    st.set_page_config(page_title = "CosmoGemini", page_icon="☁️")
    st.header('☁️ Chat with your PDFs using Cosmocloud, Google Gemini Pro & MongoDB Atlas')
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "activate_chat" not in st.session_state:
        st.session_state.activate_chat = False

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar = message['avatar']):
            st.markdown(message["content"])

    projectId = st.secrets["projectId"]
    environmentId = st.secrets["environmentId"]
    index_creation_url = st.secrets["index_creation_url"]
    search_url = st.secrets["search_url"]

    genai.configure(api_key = st.secrets['GOOGLE_AI_STUDIO_TOKEN'])

    with st.sidebar:
        st.subheader('Upload Your PDF File')
        uploaded_file = st.file_uploader('⬆️ Upload your PDF & Click to process',
                                         accept_multiple_files = False, 
                                         type=['pdf'])
        if uploaded_file is not None:
            if st.button('Process'):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                # Load the PDF
                loader = PyPDFLoader(tmp_file_path)
                data = loader.load()

                # Split PDF into documents
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
                split_docs = text_splitter.split_documents(data)
\
                query_engine = generate_index_entry(index_creation_url, 
                                                    projectId, 
                                                    environmentId, 
                                                    split_docs)

                if "query_engine" not in st.session_state:
                    st.session_state.query_engine = query_engine
                    st.session_state.activate_chat = True
                    os.unlink(tmp_file_path)

    if st.session_state.activate_chat == True or st.button('Ask'):
        if prompt := st.chat_input("Ask your question from the PDF?"):
            with st.chat_message("user", avatar='👨🏻'):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "avatar": '👨🏻', "content": prompt})

            # Make the API request to search PDF data
            context = search_pdf_data(search_url, projectId, environmentId, prompt)
            st.write(context)

            if context:
                # Generate a response using the AI model
                ai_response = generate_ai_response(context, prompt)

                # Display the assistant's response
                with st.chat_message("assistant", avatar='🤖'):
                    st.markdown(ai_response)
                st.session_state.messages.append({"role": "assistant", "avatar": '🤖', "content": ai_response})
            else:
                # Display an error if no context is found
                with st.chat_message("assistant", avatar='🤖'):
                    st.markdown("Sorry, no relevant context was found for your query.")
                st.session_state.messages.append({"role": "assistant", "avatar": '🤖', "content": "No relevant context found."})

if __name__ == '__main__':
    main()
