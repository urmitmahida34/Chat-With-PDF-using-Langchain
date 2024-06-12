# Chat-With-PDF-using-Langchain

PDF Text Analysis and Q&A
This project processes and analyzes text from PDF documents using Streamlit, PyPDF2, LangChain, FAISS, and Google's Generative AI. Key functionalities include:
- Extract Text from PDFs: Uses PyPDF2 to read and extract text from PDF documents.
- Text Chunking: Splits the extracted text into manageable chunks with a custom splitter.
- Vector Store Creation: Converts text chunks into vectors using Google Generative AI embeddings and stores them in a FAISS index.
- Conversational AI: Sets up a Q&A chain using Google's Generative AI to provide detailed answers based on the context from PDF documents.
- User Interaction: Allows users to input questions and get answers based on the PDF content.
  
To use this project, configure your Google API key in the environment variables and run the Streamlit application.
