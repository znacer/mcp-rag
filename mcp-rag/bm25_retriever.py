import bm25s
import os


class BM25Retriever:
    def __init__(self, corpus_path):
        self.corpus_path = corpus_path
        self.corpus = self._load_corpus()
        self.bm25_model = self._index_corpus()

    def _load_corpus(self):
        corpus = []
        for filename in os.listdir(self.corpus_path):
            if filename.endswith(".txt"):
                with open(
                    os.path.join(self.corpus_path, filename), "r", encoding="utf-8"
                ) as f:
                    code = f.read()
                code = code.split(sep="Article")
                code_title = code[0].splitlines()[0]
                for elt in code[1:]:
                    corpus.append(code_title + "\n" + elt)
        return corpus

    def _index_corpus(self):
        # tokenized_corpus = [bm25s.tokenize(doc, stopwords="fr") for doc in self.corpus]
        tokenized_corpus = bm25s.tokenize(self.corpus, stopwords="fr")
        bm25_model = bm25s.BM25()
        bm25_model.index(tokenized_corpus)
        return bm25_model

    def retrieve(self, query, top_n=3):
        tokenized_query = bm25s.tokenize(query)
        results, _scores = self.bm25_model.retrieve(
            tokenized_query, corpus=self.corpus, k=top_n
        )
        return results
