# Smart Medication & Care Assistant for Elderly

## Overview

A Flask-based Retrieval-Augmented Generation (RAG) application designed to help older adults and caregivers find information about medication safety and elderly care using reference documents.

## Features

- Ask care-related questions through a web interface.
- Retrieve relevant information from reference documents using FAISS.
- Generate grounded answers using Google's Gemini API.
- Handle potential emergencies with dedicated safety responses.
- Manage elderly profiles and caregiver contacts.
- Track medications and daily doses.
- Run automated tests using pytest.

## Technologies

- Python
- Flask
- Google Gemini API
- FAISS
- Sentence Transformers
- PyMuPDF
- SQLite
- HTML and CSS
- pytest

## Setup

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install the dependencies:

   `pip install -r requirements.txt`

4. Create a local `.env` file with your Gemini API key and the required configuration.
5. Add the reference PDFs and build the vector store using the project's ingestion script.
6. Start the application:

   `python app.py`

7. Open `http://127.0.0.1:5000/` in your browser.

## Run tests

`python -m pytest -v`

## Safety notice

This project is an educational care-information assistant, not a substitute for professional medical advice, diagnosis, or treatment. In an emergency, contact local emergency services.

## Privacy

Do not commit API keys, environment files, private databases, or personal medical information.
