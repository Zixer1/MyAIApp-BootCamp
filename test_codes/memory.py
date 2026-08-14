import chromadb
import chromadb.utils.embedding_functions as ef

db = chromadb.PersistentClient(path="../chroma_db")
memories = db.get_or_create_collection("my_facts")

# <<< CHANGE THESE TO THREE FACTS ABOUT YOU >>>
memories.upsert(
    documents=[
        "Menr",
        "Cool Market",
        "Class"
    ],
    ids=["fact4", "fact5", "fact6"],
)

print("\nstored:", memories.count(), "facts")

question = "Food"
results = memories.query(query_texts=[question], n_results=4)

print(results["documents"], results["distances"])