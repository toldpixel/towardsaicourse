from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)
from typing import List

class HybridRetriever(BaseRetriever):
    """Hybrid retriever that performs both semantic search and keyword search."""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        keyword_retriever: KeywordTableSimpleRetriever,
        max_retrieve: int = 10,
    ) -> None:
        """Init params."""

        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever
        self._max_retrieve = max_retrieve
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        # Run both retrievers independently so you get two ranked lists of candidate nodes—one semantic, one lexical.
        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)

        resulting_nodes = []
        node_ids_added = set()
        for i in range(min(len(vector_nodes), len(keyword_nodes))):
            vector_node = vector_nodes[i]
            if vector_node.node.node_id not in node_ids_added:
                resulting_nodes += [vector_node]
                node_ids_added.add(vector_node.node.node_id)

            keyword_node = keyword_nodes[i]
            if keyword_node.node.node_id not in node_ids_added:
                resulting_nodes += [keyword_node]
                node_ids_added.add(keyword_node.node.node_id)

        return resulting_nodes #a deduplicated, interleaved sequence of semantic and keyword retrieval hits