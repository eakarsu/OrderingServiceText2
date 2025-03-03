from llama_index.core import SimpleDirectoryReader
from llama_index.core.extractors import SummaryExtractor, QuestionsAnsweredExtractor
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.schema import TransformComponent
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
# from llama_index.readers.web import SimpleWebPageReader
from llama_index.llms.openai import OpenAI
from config import load_config, get_openai_api

# load app configuration
load_config()
print(get_openai_api())
llm = OpenAI(api_key=get_openai_api(), temperature=0.8, model="gpt-4")

class CustomTransformation(TransformComponent):
    def __call__(self, nodes, **kwargs):
        # run any node transformation logic here
        return nodes


db = chromadb.PersistentClient(path="chroma_database")
chroma_collection = db.get_or_create_collection(
    "my_chroma_store"
)

vector_store = ChromaVectorStore(
    chroma_collection=chroma_collection
)
storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)

if chroma_collection.count() > 0:
# try:
    # Rebuild the Index from the ChromaDB in future sessions
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context
    )
    print("DB found. Running using db to build index...")
else:
# except:
    cached_hashes = ""
    print("No DB found. Running without db...")

    reader = SimpleDirectoryReader('files')
    documents = reader.load_data()

    # urls = ["https://www.citifieldstadium.com"]
    # urls = ["https://alliantbenefits.cld.bz/Proofpoint-2024-Benefits-Guide/"]
    # documents_web = SimpleWebPageReader().load_data(urls)

    pipeline = IngestionPipeline(
        transformations=[
            CustomTransformation(),
            TokenTextSplitter(
                separator=" ",
                chunk_size=512,
                chunk_overlap=128),
            SummaryExtractor(),
            QuestionsAnsweredExtractor(
                questions=3
            )
        ],
        vector_store=vector_store,
        # cache=cached_hashes
    )

    nodes = pipeline.run(documents=documents, show_progress=True)
    # nodes_web = pipeline.run(documents=documents_web, show_progress=True)
    # pipeline.cache.persist("./ingestion_cache.json")


    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context
    )

print("All documents loaded")

#the following part displays the entire contents of the ChromaDB collection
results = chroma_collection.get()
print(results)