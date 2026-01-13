import asyncio
import os
import certifi
import ssl

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore


from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from logger import Colors,log_error,log_header,log_info,log_success,log_warning


load_dotenv()

# using certifi to create deafault ssl certificate file
ssl_context=ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()



# Free embedding model - HuggingFace (runs locally, no API key needed)
# Options:
# - "sentence-transformers/all-mpnet-base-v2" (768 dims, better quality) - CURRENT
# - "sentence-transformers/all-MiniLM-L6-v2" (384 dims, fast)
# - "sentence-transformers/all-MiniLM-L12-v2" (384 dims, balanced)
# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-mpnet-base-v2"
# )

# vectorstore=PineconeVectorStore(index_name=os.environ.get("PINECONE_INDEX_NAME"),embedding=embeddings)

# tavily_extract = TavilyExtract()
# tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()






print("Ran")

async def main():
    log_header("DOCUMENTATION INGESTION PIPELINE.....")


    log_info(message="Crawling from langchain document...",
    color=Colors.PURPLE)
    tavily_crawl_result=tavily_crawl.invoke(input={
        "url": "https://python.langchain.com/",
        "max_depth":5,
        "extract_depth":"advanced",
        "instructions":"relevant to ai agents"
    })
    # print(tavily_crawl_result)
    all_docs=[Document(page_content=result['raw_content'],metadata={"source":result['url']}) for result in tavily_crawl_result["results"]]

    # print(len(all_docs))
    log_success("CRAWLING COMLETED")

    log_header("CHUNKING THE LOADED DOCUMENTS....")

    text_splitter=RecursiveCharacterTextSplitter(chunk_size=4000,chunk_overlap=200)

    splitted_docs=text_splitter.split_documents(all_docs)

    print(len(splitted_docs))




if __name__=="__main__":
    asyncio.run(main())
