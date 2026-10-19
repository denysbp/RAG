*This project has been created as part of the 42 curriculum by deferrei.*

# RAG

## Description

RAG is a project focused on implementing a Retrieval-Augmented Generation (RAG) retrieval pipeline. Its purpose is to retrieve relevant information from a collection of documents in response to a user query.

The project explores different information retrieval techniques and compares their results. It currently includes two retrieval approaches:

- **BM25:** a lexical retrieval method that ranks documents according to term matching and relevance.
- **Semantic Search:** a vector-based retrieval method that represents text and queries as embeddings and ranks results according to semantic similarity.

The project also includes an indexing workflow that prepares documents and stores their processed representations for retrieval. The next step is to combine the results from different retrieval methods using Reciprocal Rank Fusion (RRF).

The main objective is to understand how different retrieval strategies work, how their results differ, and how they can be combined to improve retrieval quality.

## Instructions

### Full pipeline

```
1 — index the VLLM repository
uv run python -m src index

2 — search a single question
uv run python -m src search "How does the scheduler work in vLLM?" --k 5

3 — answer a single question
uv run python -m src answer "What is PagedAttention?" --k 5

4 — search an entire dataset (docs)
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 5 --save_directory data/output/search_results

4.1 — search an entire dataset (code)
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json \
    --k 5 --save_directory data/output/search_results

5 — evaluate retrieval quality (docs)
uv run python -m src evaluate \
    data/output/search_results/dataset_docs_public.json \
    data/datasets/AnsweredQuestions/dataset_docs_public.json --k 5

5.1 — evaluate retrieval quality (code)
uv run python -m src evaluate \
    data/output/search_results/dataset_code_public.json \
    data/datasets/AnsweredQuestions/dataset_code_public.json --k 5

6 — generate answers for a dataset (docs)
uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer

6.1 — generate answers for a dataset (code)
uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/dataset_code_public.json \
    --save_directory data/output/search_results_and_answer

```

### Requirements

The project uses Python and `uv` for dependency and environment management.

The semantic retrieval component uses the following technologies:

- Python
- NumPy
- Sentence Transformers
- `all-MiniLM-L6-v2`
- `uv`

If you don't have `uv`, you can install with `make install`.

### Installation

Clone the repository:

```bash
git clone https://github.com/denysbp/RAG.git
cd RAG
```

Install the project dependencies:

```bash
uv sync
```

### Indexing

Before performing searches, index the document collection using the indexing command implemented by the project.

```bash
uv run -m src index --max_chunk_size <int>
```

The indexing process prepares the documents and creates the data required by the retrieval methods.

### Searching with BM25

Run a lexical search with a query and specify the number of results to retrieve:

```bash
uv run -m src search --query "What is vllm" --k 5
```

The command retrieves the top five results ranked by BM25.

### Searching with Semantic Search

Run a semantic search using the same query:

```bash
uv run -m src semantic_search --query "What is vllm" --k 5
```

The semantic search component encodes the query using `all-MiniLM-L6-v2` and compares it with the stored document embeddings.

The `--k` parameter specifies the maximum number of results to return.

### Comparing Retrieval Methods

Run both commands with the same query and compare the retrieved sources. BM25 focuses on lexical matching, while semantic search focuses on similarity between vector representations.

The results may differ because the two methods use different approaches to estimate relevance.

## Resources

### Technical References

- [Sentence Transformers](https://www.sbert.net/) -  Documentation for generating sentence and text embeddings.
- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) - The embedding model used by the semantic retrieval component.
- [NumPy](https://numpy.org/doc/) - Numerical operations and vector similarity calculations.
- [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) - Background on the lexical ranking algorithm.
- [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) - Research paper introducing the RRF ranking method.
- some youtubes videos too.

### AI Usage

AI tools were used exclusively to improve the English wording and readability of this README. They were not used to develop, generate, or implement the project's code. The project's implementation, technical decisions, and development were carried out by the legend _me_.

## System Architecture

The project follows a retrieval pipeline that prepares a document collection and uses the resulting index to answer search queries.

The main stages are:

1. **Document loading:** the project loads the available document collection.
2. **Document processing:** documents are processed and divided into chunks suitable for retrieval.
3. **Indexing:** the processed data is prepared for use by the retrieval methods.
4. **Embedding generation:** the semantic retrieval component represents the processed text as numerical vectors using `all-MiniLM-L6-v2`.
5. **Query processing:** the user provides a query and specifies how many results should be returned.
6. **Retrieval:** BM25 or semantic search ranks the available sources.
7. **Result selection:** the top `k` sources are returned to the user.

The two retrieval methods use different relevance signals. BM25 relies on lexical information, while semantic search compares the query embedding with the stored embeddings.

Reciprocal Rank Fusion is intended to combine the rankings produced by these methods into a single ranking.

## Chunking Strategy

Documents are divided into smaller chunks before retrieval. Chunking allows the system to search for relevant portions of a document rather than treating every document as a single unit.

The resulting chunks are used by the retrieval pipeline and, for semantic search, are represented by embeddings.

The exact chunk size, overlap, and splitting rules depend on the chunking implementation in the repository.

**Implementation details to document:**
- Chunk size: `The amount of chars per chunk`
- Chunk overlap: `0`
- Splitting strategy: `I use 3 strategy for .txt, .py, .md`

These values should reflect the actual implementation rather than assumed defaults.

## Retrieval Methods and Ranking

### BM25

BM25 is a lexical information retrieval algorithm. It ranks sources according to query-term matches, taking into account term frequency, document length, and the distribution of terms across the collection.

It is useful when the query contains specific words, technical terms, identifiers, or names that also appear in the indexed text.

### Semantic Search

Semantic search uses text embeddings to represent queries and indexed chunks as numerical vectors.

The project uses `all-MiniLM-L6-v2` to encode the query. It then compares the query embedding with the stored embeddings using a matrix multiplication:

![Formula](https://miro.medium.com/v2/resize:fit:640/format:webp/0*7dWSG0979AvY4Wo2)

```python
similarities = embeddings @ query_embedding
```

The similarity scores are sorted in descending order, and the highest-ranked sources are selected.

When the embeddings are normalized, their dot product is equivalent to cosine similarity. This makes it possible to rank sources according to the similarity between their vector representations.

Semantic search can retrieve text that is conceptually related to a query even when it does not share exactly the same words.

### Reciprocal Rank Fusion (RRF)

Reciprocal Rank Fusion is intended to combine the rankings from BM25 and semantic search.

Instead of directly comparing the raw scores produced by different retrieval algorithms, RRF uses the position of each source in each ranking.

A common RRF formula is:

![rff formula](https://miro.medium.com/v2/resize:fit:4800/format:webp/1*TSji0q0ZO1JBqxBaKPA28w.png)


Sources that rank well across multiple retrieval methods can receive a higher combined score.

The implementation and evaluation of RRF should be documented once the method is completed and tested.

## Performance Analysis

Retrieval quality should be evaluated using a labelled set of queries with known relevant sources.

### Recall@k

Recall@k measures the proportion of relevant sources that appear in the top `k` retrieved results.

Recall@k = (Number of queries with target in top-k) / (Total queries)

For example, if a query has four relevant sources and the top five results contain three of them, its Recall@5 is:

**Recall@5 = 3 / 4 = 0.75 (75%)**

This means that 75% of the known relevant sources were retrieved within the first five results.

### Evaluation Results

The following table can be completed after running the evaluation on the same query set for each retrieval method.

| Data type | Recall@1 | Recall@3 | Recall@5 |
|---|---:|---:|---:|
| docs | 50.0% | 74.0% | 81.0% |
| code | 26.3% | 48.5% | 56.6 |

### Performance Considerations

The retrieval methods have different computational characteristics:

- **BM25** performs lexical ranking and depends on the implementation and size of the indexed collection.
- **Semantic Search** requires embedding generation for queries and similarity calculations against stored embeddings. Precomputing document embeddings avoids re-encoding every chunk for every query.
- **RRF** combines ranked lists using rank positions, making it possible to merge different retrieval methods without requiring their raw scores to be on the same scale.

Actual latency and resource consumption should be measured on the target dataset and hardware before drawing conclusions about which method is faster.

## Design Decisions

### Separate Retrieval Methods

BM25 and semantic search are implemented as separate retrieval approaches so that their results can be tested and compared independently.

This separation also makes it possible to investigate hybrid retrieval without changing the internal ranking logic of each method.

### Precomputed Embeddings

Document embeddings are stored and reused rather than being regenerated for every search. This reduces repeated computation during semantic retrieval.

The query embedding is generated at search time because the query changes from one request to another.

### NumPy for Similarity Calculations

NumPy is used to calculate similarity scores between the query embedding and the stored document embeddings. Matrix operations provide a straightforward way to compare a query against the indexed collection.

### Top-k Retrieval

The `k` parameter controls the number of results returned. Validating this parameter helps prevent invalid requests, such as asking for zero results or requesting more results than the index contains.

### Hybrid Retrieval

RRF is being explored as a way to combine lexical and semantic rankings. It offers a ranking-based approach that does not require directly comparing BM25 scores with embedding similarity scores.

## Challenges

The project explores several practical challenges in information retrieval:

- **Different ranking behaviours:** BM25 and semantic search can return different sources for the same query because they measure relevance differently.
- **Project Pipeline:** I passed the first day only trying undertand how the project is suposed to work.

## Bonus Features

### 1. Caching

Implemented a caching mechanism to avoid unnecessary computations and improve performance.

- **Index caching:** Reuses the existing index when no changes are detected and the chunk size remains the same.
- **Query caching:** Stores search results for previously executed queries, avoiding repeated retrieval operations for the same query and parameters.

### 2. Incremental Indexing

Implemented incremental indexing to avoid rebuilding the entire index whenever the source files change.

- Detects new, modified, and deleted files.
- Updates the index based on the detected changes.
- Preserves existing indexed content whenever possible, reducing unnecessary processing.

### 3. HTTP API

Implemented an HTTP interface to expose the application's functionality through HTTP endpoints, allowing external clients to interact with the system without invoking the CLI directly.

### 4. Hybrid Search

Implemented hybrid retrieval by combining BM25 lexical search and semantic search using Reciprocal Rank Fusion (RRF).

The evaluation achieved the following Recall@5 scores:

- **Documentation dataset:** 81.0% (requirement: at least 80%).
- **Code dataset:** 56.6% (requirement: at least 50%).

Both evaluation requirements were satisfied.
