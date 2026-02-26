from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")



def ranker(query,retrieved_paragraphs):
    input_pairs = [(query, paragraph) for paragraph in retrieved_paragraphs]
    scores = model.predict(input_pairs)
    ranked = sorted(zip(retrieved_paragraphs, scores), key=lambda x: x[1], reverse=True)
    top_paragraphs = [i[0] for i in ranked[:5]]
    return top_paragraphs
