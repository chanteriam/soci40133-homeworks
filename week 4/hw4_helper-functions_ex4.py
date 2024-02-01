import numpy as np
import copy
import gensim
import pandas as pd
import sklearn.metrics.pairwise


def intersection_align_gensim(m1, m2, words=None):
    """
    Intersect two gensim word2vec models, m1 and m2.
    Only the shared vocabulary between them is kept.
    If 'words' is set (as list or set), then the vocabulary is intersected with this list as well.
    Indices are re-organized from 0..N in order of descending frequency (=sum of counts from both m1 and m2).
    These indices correspond to the new syn0 and syn0norm objects in both gensim models:
        -- so that Row 0 of m1.syn0 will be for the same word as Row 0 of m2.syn0
        -- you can find the index of any word on the .index2word list: model.index2word.index(word) => 2
    The .vocab dictionary is also updated for each model, preserving the count but updating the index.
    """

    # Get the vocab for each model
    vocab_m1 = set(m1.wv.index_to_key)
    vocab_m2 = set(m2.wv.index_to_key)

    # Find the common vocabulary
    common_vocab = vocab_m1 & vocab_m2
    if words:
        common_vocab &= set(words)

    # If no alignment necessary because vocab is identical...
    if not vocab_m1 - common_vocab and not vocab_m2 - common_vocab:
        return (m1, m2)

    # Otherwise sort by frequency (summed for both)
    common_vocab = list(common_vocab)
    common_vocab.sort(
        key=lambda w: m1.wv.get_vecattr(w, "count") + m2.wv.get_vecattr(w, "count"),
        reverse=True,
    )

    # Then for each model...
    for m in [m1, m2]:
        # Replace old syn0norm array with new one (with common vocab)
        new_arr = [m.wv.get_vector(w, norm=True) for w in common_vocab]

        # Replace old vocab dictionary with new one (with common vocab)
        # and old index2word with new one
        m.index2word = common_vocab
        # old_vocab = m.wv.index_to_key
        new_vocab = []
        k2i = {}
        for new_index, word in enumerate(common_vocab):
            new_vocab.append(word)
            k2i[word] = new_index
        m.wv.index_to_key = new_vocab
        m.wv.key_to_index = k2i
        m.wv.vectors = np.array(new_arr)

    return (m1, m2)


def smart_procrustes_align_gensim(base_embed, other_embed, words=None):
    base_embed = copy.copy(base_embed)
    other_embed = copy.copy(other_embed)
    # make sure vocabulary and indices are aligned
    in_base_embed, in_other_embed = intersection_align_gensim(
        base_embed, other_embed, words=words
    )

    # get the embedding matrices
    # base_vecs = calc_syn0norm(in_base_embed)
    # other_vecs = calc_syn0norm(in_other_embed)
    base_vecs = [
        in_base_embed.wv.get_vector(w, norm=True)
        for w in set(in_base_embed.wv.index_to_key)
    ]
    other_vecs = [
        in_other_embed.wv.get_vector(w, norm=True)
        for w in set(in_other_embed.wv.index_to_key)
    ]

    # just a matrix dot product with numpy
    m = np.array(other_vecs).T.dot(np.array(base_vecs))
    # SVD method from numpy
    u, _, v = np.linalg.svd(m)
    # another matrix operation
    ortho = u.dot(v)
    # Replace original array with modified one
    # i.e. multiplying the embedding matrix (syn0norm)by "ortho"
    other_embed.wv.vectors = (np.array(other_vecs)).dot(ortho)
    return other_embed


def rawModels(df, category, text_column_name="normalized_sents", sort=True):
    embeddings_raw = {}
    cats = sorted(set(df[category]))
    for cat in cats:
        # This can take a while
        print("Embedding {}".format(cat), end="\r")
        subsetDF = df[df[category] == cat]
        # You might want to change the W2V parameters
        embeddings_raw[cat] = gensim.models.word2vec.Word2Vec(
            subsetDF[text_column_name].sum()
        )
    return embeddings_raw


def compareModels(df, category, text_column_name="normalized_sents", sort=True):
    """Prepare embeddings and align them."""
    # Generate raw embeddings
    embeddings_raw = rawModels(df, category, text_column_name, sort)
    cats = sorted(set(df[category]))

    # Align embeddings
    embeddings_aligned = {}
    base_cat = cats[0]  # Use the first category as the base for alignment
    for cat in cats:
        if cat == base_cat:
            embeddings_aligned[cat] = embeddings_raw[cat]
        else:
            # Align current category's embedding to the base category's embedding
            embeddings_aligned[cat] = smart_procrustes_align_gensim(
                embeddings_raw[base_cat], embeddings_raw[cat]
            )
    return embeddings_raw, embeddings_aligned


def getDivergenceDF(word, embeddingsDict):
    # """Calculate pairwise divergence for a specific word across different embeddings."""
    cats = sorted(embeddingsDict.keys())  # Sorted categories, e.g., congress numbers
    # Initialize a square DataFrame with zeros
    df = pd.DataFrame(np.zeros((len(cats), len(cats))), index=cats, columns=cats)

    for i, cat_outer in enumerate(cats):
        for j, cat_inner in enumerate(cats):
            if i == j:
                # Divergence with itself should be 0
                df.iloc[i, j] = 0
            else:
                # Calculate cosine similarity for different congress embeddings
                vector_outer = embeddingsDict[cat_outer].wv[word]
                vector_inner = embeddingsDict[cat_inner].wv[word]
                similarity = sklearn.metrics.pairwise.cosine_similarity(
                    np.expand_dims(vector_outer, axis=0),
                    np.expand_dims(vector_inner, axis=0),
                )[0, 0]
                divergence = np.abs(1 - similarity)
                df.iloc[i, j] = divergence

    return df


def verify_embeddings(embeddingsDict, word):
    """Check if any embedding vector for 'word' is invalid or significantly different."""
    for cat, model in embeddingsDict.items():
        if word not in model.wv:
            print(f"Warning: '{word}' not found in embeddings for category '{cat}'.")
        else:
            vector = model.wv[word]
            if np.all(vector == 0):
                print(
                    f"Warning: Embedding vector for '{word}' in category '{cat}' is all zeros."
                )


rawEmbeddings, comparedEmbeddings = compareModels(
    congress_df, "congress_num"
)  # replace with your dataframe and category
rawEmbeddings.keys()
pltDF = getDivergenceDF("abortion", comparedEmbeddings)
pltDF
