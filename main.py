from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

file_path = "./nodejs.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()
print(docs[5])

