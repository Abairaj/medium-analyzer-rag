import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

print("Initializing components")

embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="qwen3:8b", temperature=0.3)

vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
    
    {context}
    
    Question: {question}
    
    Provide a detailed answer:
    """
)


def format_docs(docs):
    """Format retrieved documents into a single string."""

    return "\n\n".join(doc.page_content for doc in docs)


def create_retrieval_chain_with_lcel():

    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return retrieval_chain


if __name__ == "__main__":
    print("Retrieving...")

    query = "what is pinecone in machine learning ?"

    chain = create_retrieval_chain_with_lcel()
    result = chain.invoke({"question": query})
    print(result)
