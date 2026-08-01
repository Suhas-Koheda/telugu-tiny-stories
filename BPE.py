from collections import Counter
import re

class BPE:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.merges = []
        self.token2id = {}
        self.id2token = {}

    def _word_freq(self, text):
        words = re.findall(r"\S+", text.lower())
        return Counter(words)

    def train(self, text):
        word_freq = self._word_freq(text)
        vocab = {" ".join(list(word)) + " </w>": freq for word, freq in word_freq.items()}
        symbols = set()
        for word in vocab:
            symbols.update(word.split())
        while len(symbols) < self.vocab_size:
            pairs = Counter()
            for word, freq in vocab.items():
                tokens = word.split()
                for i in range(len(tokens) - 1):
                    pairs[(tokens[i], tokens[i + 1])] += freq
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.merges.append(best)
            pattern = re.compile(r"(?<!\S)" + re.escape(best[0] + " " + best[1]) + r"(?!\S)")
            new_vocab = {}
            for word, freq in vocab.items():
                new_word = pattern.sub(best[0] + best[1], word)
                new_vocab[new_word] = freq
            vocab = new_vocab
            symbols = set()
            for word in vocab:
                symbols.update(word.split())
        tokens = sorted(symbols)
        self.token2id = {tok: i for i, tok in enumerate(tokens)}
        self.id2token = {i: tok for tok, i in self.token2id.items()}

    def encode(self, text):
        ids = []
        words = re.findall(r"\S+", text.lower())
        for word in words:
            tokens = list(word) + ["</w>"]
            for a, b in self.merges:
                i = 0
                merged = []
                while i < len(tokens):
                    if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                        merged.append(a + b)
                        i += 2
                    else:
                        merged.append(tokens[i])
                        i += 1
                tokens = merged
            ids.extend(self.token2id[t] for t in tokens if t in self.token2id)
        return ids

    def decode(self, ids):
        text = ""
        for idx in ids:
            tok = self.id2token[idx]
            if tok == "</w>":
                text += " "
            elif tok.endswith("</w>"):
                text += tok[:-4] + " "
            else:
                text += tok
        return text.strip()


# text = """
# Byte Pair Encoding is a simple subword tokenization algorithm.
# It starts with characters and repeatedly merges the most frequent pair.
# Byte Pair Encoding works well for transformer models.
# """

# bpe = BPE(vocab_size=100)
# bpe.train(text)

# encoded = bpe.encode("Byte Pair Encoding works well")
# decoded = bpe.decode(encoded)

# print(encoded)
# print(decoded)
# print(len(bpe.token2id))