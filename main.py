from openai import OpenAI

from src.pdf_reader import extract_text
from src.embeddings import (
    create_chunks,
    create_embeddings,
    get_model
)

from src.search import (
    build_index,
    retrieve
)

# STEP 1
text = extract_text(
    "data/company_handbook.pdf"
)

# STEP 2
chunks = create_chunks(text)

# STEP 3
embeddings = create_embeddings(
    chunks
)

# STEP 4
index = build_index(
    embeddings
)

# STEP 5
query = input(
    "Ask your question: "
)

model = get_model()

query_embedding = model.encode(
    [query]
)

results = retrieve(
    index,
    query_embedding,
    top_k=3
)

context = ""

for idx in results:

    context += chunks[idx]
    context += "\n"

# STEP 6
client = OpenAI(
    api_key="YOUR_OPENAI_API_KEY"
)

prompt = f"""
You are an enterprise assistant.

Answer ONLY using the
provided context.

Context:
{context}

Question:
{query}
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role":"user",
            "content":prompt
        }
    ]
)

print("\nAnswer:\n")

print(
    response
    .choices[0]
    .message
    .content
)
