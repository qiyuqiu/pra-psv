"""Four causal views and 44 numeric features, using an explicit frozen encoder."""
from .core import features
from .features import make_views


class StateEncoder:
    def __init__(self, encoder, external_tools):
        self.encoder = encoder
        self.external_tools = frozenset(external_tools)
        self.cache = {}

    def pack(self, query, call, messages, runtime, score, count, tokens, embedding, used):
        views, extra, coverage = make_views(
            self.encoder.tokenizer, query, call, messages,
            runtime.functions[call.function].description, self.external_tools, used)
        vectors = []
        for text in views:
            if text not in self.cache:
                self.cache[text] = self.encoder.encode([text], batch_size=1)[0].tolist()
            vectors.append(self.cache[text])
        base = features(query, call, messages, score, count, tokens,
                        self.external_tools, embedding)[:12]
        return dict(numeric=base + extra, views=vectors, coverage=coverage, pra_block=score >= .89)

