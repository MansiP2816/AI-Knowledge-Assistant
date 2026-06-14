from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

def create_chunks(text, chunk_size=500):

    chunks = []

    for i in range(0, len(text), chunk_size):

        chunks.append(
            text[i:i+chunk_size]
        )

    return chunks


def create_embeddings(chunks):

    return model.encode(chunks)


def get_model():

    return model
