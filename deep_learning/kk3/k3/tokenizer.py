from transformers import PreTrainedTokenizer


class ByteLevelTokenizer(PreTrainedTokenizer):
    model_input_names = ["input_ids"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def vocab_size(self):
        return 256

    def get_vocab(self):
        return {chr(i): i for i in range(256)}

    def _tokenize(self, text, **kwargs):
        return [chr(b) for b in text.encode("utf-8")]

    def _convert_token_to_id(self, token):
        return ord(token) if len(token) == 1 else 0

    def _convert_id_to_token(self, index):
        return chr(index)

    def convert_tokens_to_string(self, tokens):
        try:
            return "".join(tokens).encode("latin1").decode("utf-8", errors="replace")
        except:
            return "".join(tokens)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        return tuple()
