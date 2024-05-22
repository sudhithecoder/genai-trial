# Project Setup Guide

## Prerequisites

1. Get your OpenAPI key.
2. Obtain your Langchain API key.

## Update Environment Variables

Update the `.env` file with the obtained API keys:


OPENAPI_KEY=<your_openapi_key>
LANGCHAIN_API_KEY=<your_langchain_api_key>


## Install Dependencies

Install the necessary dependencies by running the following command:
pip install -r requirements.txt


## Running the ETL Process

The `retriever.py` file contains all the ETL (Extract, Transform, Load) logic, including tokenizing and embedding. You can run this file separately by adding a print statement and executing it with:
python3 retriever.py


## Running the Streamlit App

To run the Streamlit app, use the following command:
streamlit run client.py
