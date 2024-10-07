# CosmoGemini ☁️
## Chat with your PDFs using Cosmocloud, Google Gemini Pro & MongoDB Atlas

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone this repository:
   ```
   https://github.com/adu0502/cosmo-pdf-chat.git
   cd cosmo-pdf-chat
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Requirements

The following packages are required for this application:

- streamlit==1.39.0
- langchain-community==0.3.1
- google-generativeai==0.8.2
- llama-index
- pypdf
- requests

These requirements are listed in the `requirements.txt` file.

## Running the Application

To run the Streamlit application, use the following command:

```
streamlit run pdf-q-a-streamlit-app.py
```
