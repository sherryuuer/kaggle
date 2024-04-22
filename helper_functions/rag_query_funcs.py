# define helper function to print wrapped text
from time import perf_counter as timer
import textwrap


def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)


# 这段代码定义了一个名为 `print_wrapped` 的辅助函数，其功能是打印被包装过的文本。
# 具体来说，它的作用是将输入的文本按照指定的行宽进行自动换行，并打印包含换行后的文本。

# 函数的输入包括两个参数：

# 1. `text`：要打印的文本内容。
# 2. `wrap_length`：行宽，即每行文本的最大字符数，默认为 80。

# 函数首先使用 `textwrap.fill()` 函数将输入的文本 `text` 按照指定的行宽进行包装处理，得到一个包含换行符的文本字符串 `wrapped_text`。
# 然后，它通过 `print()` 函数将包装后的文本打印到控制台上。

# 这个函数的作用在于使得长文本在打印时能够自动换行，以保持输出的整洁和易读。


def retrieve_relevant_resources(
    query: str,
    embeddings: torch.tensor,
    model: SentenceTransformer = embedding_model,
    n_resources_to_return: int = 5,
    print_time: bool = True
):
    """
    Embeds a query with model and returns top k scores and indices from embeddings.
    """
    # embed the query
    query_embedding = model.encode(query, convert_to_tensor=True)

    # get dot product scores on embeddings
    start_time = timer()
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    end_time = timer()

    if print_time:
        print(f"[INFO] Time taken to get scores on {
              len(embeddings)} embeddings: {end_time - start_time:.5f} seconds.")

    scores, indices = torch.topk(
        input=dot_scores,
        k=n_resources_to_return
    )
    return scores, indices


def print_top_results_and_scores(query: str,
                                 embeddings: torch.tensor,
                                 pages_and_chunks: list[dict] = pages_and_chunks,
                                 n_resources_to_return: int = 5):
    """
    Takes a query, retrieves most relevant resources and prints them out in descending order.

    Note: Requires pages_and_chunks to be formatted in a specific way (see above for reference).
    """

    scores, indices = retrieve_relevant_resources(query=query,
                                                  embeddings=embeddings,
                                                  n_resources_to_return=n_resources_to_return)

    print(f"Query: {query}\n")
    print("Results:")
    # Loop through zipped together scores and indicies
    for score, index in zip(scores, indices):
        print(f"Score: {score:.4f}")
        # Print relevant sentence chunk (since the scores are in descending order, the most relevant chunk will be first)
        print_wrapped(pages_and_chunks[index]["sentence_chunk"])
        # Print the page number too so we can reference the textbook further and check the results
        print(f"Page number: {pages_and_chunks[index]['page_number']}")
        print("\n")


# test the functions
query = "symptoms of pellagra"
# get the score and indices
score, indices = retrieve_relevent_resources(
    query=query, embeddings=embeddings)
# print out the texts of the top scores
print_top_results_and_scores(query=query, embeddings=embeddings)
