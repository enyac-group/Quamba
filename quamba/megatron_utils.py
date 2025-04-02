# The code is extracted and modified from
# https://github.com/NVIDIA/Megatron-LM/blob/core_r0.10.0/megatron/training/tokenizer/tokenizer.py
#
# NOTE(hychiang): Special handle for mamba2-8b's tokenizer from NVIDIA Megatron
# https://github.com/NVIDIA/Megatron-LM/issues/864
# https://github.com/NVIDIA/Megatron-LM/blob/core_r0.10.0/megatron/training/tokenizer/tokenizer.py
# https://github.com/NVIDIA/Megatron-LM/blob/core_r0.10.0/examples/mamba/run_text_gen_server_8b.sh#L42
# https://github.com/NVIDIA/Megatron-LM/blob/core_r0.10.0/megatron/training/arguments.py#L1791
import os
import torch
import sentencepiece
from sentencepiece import sentencepiece_model_pb2 as spm_pb2
from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer


class _SentencePieceTokenizer(MegatronTokenizer):
    """SentencePieceTokenizer-Megatron wrapper"""

    def __init__(self, model_file, vocab_extra_ids=0):
        super().__init__(model_file, vocab_extra_ids=vocab_extra_ids)
        self.model_file = model_file
        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=model_file)
        self._initalize(vocab_extra_ids)

    def _populate_vocab(self):
        self._vocab = {}
        self._inv_vocab = {}

        for i in range(len(self.tokenizer)):
            t = self.tokenizer.id_to_piece(i)
            self._inv_vocab[i] = t
            self._vocab[t] = i

    def _initalize(self, vocab_extra_ids):
        self._populate_vocab()
        self._special_tokens = {}
        self._inv_special_tokens = {}

        self._t5_tokens = []

        def _add_special_token(t):
            if t not in self._vocab:
                next_id = len(self._vocab)
                self._vocab[t] = next_id
                self._inv_vocab[next_id] = t
            self._special_tokens[t] = self._vocab[t]
            self._inv_special_tokens[self._vocab[t]] = t

        _add_special_token('<CLS>')
        self._cls_id = self._vocab['<CLS>']
        _add_special_token('<SEP>')
        self._sep_id = self._vocab['<SEP>']
        _add_special_token('<EOD>')
        self._eod_id = self._vocab['<EOD>']
        _add_special_token('<MASK>')
        self._mask_id = self._vocab['<MASK>']

        pad_id = self.tokenizer.pad_id()
        try:
            pad_token = self.tokenizer.id_to_piece(pad_id)
        except IndexError:
            pad_token = '<PAD>'
        _add_special_token(pad_token)
        self._pad_id = self._vocab[pad_token]

        bos_id = self.tokenizer.bos_id()
        try:
            bos_token = self.tokenizer.id_to_piece(bos_id)
        except IndexError:
            bos_token = '<BOS>'
        _add_special_token(bos_token)
        self._bos_id = self._vocab[bos_token]

        eos_id = self.tokenizer.eos_id()
        try:
            eos_token = self.tokenizer.id_to_piece(eos_id)
        except IndexError:
            eos_token = '<EOS>'
        _add_special_token(eos_token)
        self._eos_id = self._vocab[eos_token]

        for i in range(vocab_extra_ids):
            t = "<extra_id_{}>".format(i)
            _add_special_token(t)
            self._t5_tokens += [t]

    @property
    def vocab_size(self):
        return len(self._vocab)

    @property
    def vocab(self):
        return self._vocab

    @property
    def inv_vocab(self):
        return self._inv_vocab

    @property
    def decoder(self):
        return self._inv_vocab

    @property
    def encoder(self):
        return self._vocab

    # From:
    # https://github.com/NVIDIA/NeMo/blob/c8fa217e811d60d11d014827c7f3845ff6c99ae7/nemo/collections/common/tokenizers/sentencepiece_tokenizer.py#L89
    def tokenize(self, text):
        ids = []
        idx = 0

        while 1:
            indices = {}
            for token in self._special_tokens:
                try:
                    indices[token] = text[idx:].index(token)
                except ValueError:
                    continue
            if len(indices) == 0:
                break

            next_token = min(indices, key=indices.get)
            next_idx = idx + indices[next_token]

            ids.extend(self.tokenizer.encode_as_ids(text[idx:next_idx]))
            ids.append(self._special_tokens[next_token])
            idx = next_idx + len(next_token)

        ids.extend(self.tokenizer.encode_as_ids(text[idx:]))
        return ids

    # From:
    # https://github.com/NVIDIA/NeMo/blob/c8fa217e811d60d11d014827c7f3845ff6c99ae7/nemo/collections/common/tokenizers/sentencepiece_tokenizer.py#L125
    def detokenize(self, ids):
        text = ""
        last_i = 0

        for i, id in enumerate(ids):
            if id in self._inv_special_tokens:
                text += self.tokenizer.decode_ids(ids[last_i:i]) + " "
                text += self._inv_special_tokens[id] + " "
                last_i = i + 1

        text += self.tokenizer.decode_ids(ids[last_i:])
        return text

    def offsets(self, ids: list[int], text: str) -> list[int]:
        return [p.begin for p in self.tokenizer.decode_ids_as_immutable_proto(ids).pieces]

    @property
    def cls(self):
        return self._cls_id

    @property
    def sep(self):
        return self._sep_id

    @property
    def pad(self):
        return self._pad_id

    @property
    def bos(self):
        return self._bos_id

    @property
    def eod(self):
        return self._eod_id

    @property
    def eos(self):
        return self._eos_id

    @property
    def mask(self):
        return self._mask_id

    @property
    def additional_special_tokens_ids(self):
        return [self.vocab[k] for k in self._t5_tokens]
    
    # https://github.com/google/sentencepiece/issues/387
    # https://github.com/google/sentencepiece/issues/121
    def save(self, save_dir):
        # read the model in protobuf format
        m = spm_pb2.ModelProto()
        m.ParseFromString(open(self.model_file, 'rb').read())
        # and save
        save_path = os.path.join(save_dir, os.path.basename(self.model_file))
        with open(save_path, 'wb') as f:
            f.write(m.SerializeToString())
    

class _GPTSentencePieceTokenizer(_SentencePieceTokenizer):
    """SentencePieceTokenizer-Megatron wrapper"""

    def __init__(self, model_file):
        super().__init__(model_file, vocab_extra_ids=0)

    def _initalize(self, vocab_extra_ids):
        self._populate_vocab()

        self._pad_id = self.tokenizer.pad_id()
        self._bos_id = self.tokenizer.bos_id()
        self._eos_id = self.tokenizer.eos_id()

    def tokenize(self, text):
        return self.tokenizer.encode_as_ids(text)

    def detokenize(self, ids):
        return self.tokenizer.decode_ids(ids)

    @property
    def cls(self):
        return -1

    @property
    def sep(self):
        return -1

    @property
    def mask(self):
        return -1

    @property
    def eod(self):
        return self._eos_id

    @property
    def additional_special_tokens_ids(self):
        return None

class Token:

    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask


class _GPTSentencePieceTokenizer(_SentencePieceTokenizer):
    """SentencePieceTokenizer-Megatron wrapper"""

    def __init__(self, model_file, padding_side = "left"):
        super().__init__(model_file, vocab_extra_ids=0)
        self.padding_side = "left"

    def _initalize(self, vocab_extra_ids):
        self._populate_vocab()

        self._pad_id = self.tokenizer.pad_id()
        self._bos_id = self.tokenizer.bos_id()
        self._eos_id = self.tokenizer.eos_id()

    # To unified the APIs with huggingface, we add a __call__ function
    def __call__(self, text, return_tensors="pt", padding="longest", max_length=None, truncation=True, add_special_tokens=False):
        if isinstance(text, str):
            input_ids = self.tokenizer.encode_as_ids(text)
            if max_length is not None and max_length>0:
                input_ids = input_ids[:max_length]
            if return_tensors=="pt":
                input_ids = torch.tensor(input_ids, dtype=torch.long)
                if len(input_ids.shape) == 1:
                    input_ids = input_ids.unsqueeze(0) # shape: [seqlen] -> [1, seqlen]
            return Token(input_ids, torch.ones_like(input_ids))
        # This is a hack to accommodate the huggingface's tokenizer API for testing mamba2-8b on the generation tasks
        elif isinstance(text, (list, tuple)):
            input_ids = []
            encoding = {"input_ids": [], "attention_mask": []}
            longest_seqlen = 0
            for t in text:
                ids = self.tokenizer.encode_as_ids(t)
                if max_length is not None and max_length>0:
                    ids = ids[:max_length]
                if longest_seqlen < len(ids):
                    longest_seqlen = len(ids)
                if return_tensors=="pt":
                    ids = torch.tensor(ids, dtype=torch.long)
                    if len(ids.shape) == 1:
                        ids = ids.unsqueeze(0) # shape: [seqlen] -> [1, seqlen]
                token = Token(ids, torch.ones_like(ids))
                encoding["input_ids"].append(token.input_ids)
                encoding["attention_mask"].append(token.attention_mask)
            if padding == "longest":
                for i in range(len(encoding["input_ids"])):
                    if isinstance(encoding["input_ids"][i], list):
                        pad_seq = [self._pad_id] * longest_seqlen - encoding["input_ids"][i].shape[1]
                        pad_attn = [0] * longest_seqlen - encoding["input_ids"][i].shape[1]
                        if self.padding_side == "left":
                            encoding["input_ids"][i] = pad_seq.extend(encoding["input_ids"][i])
                            encoding["attention_mask"][i] = pad_attn.extend(encoding["attention_mask"][i])
                        elif self.padding_side == "right":
                            encoding["input_ids"][i] = encoding["input_ids"][i].extend(pad_seq)
                            encoding["attention_mask"][i] = encoding["attention_mask"][i].extend(pad_attn)
                    elif isinstance(encoding["input_ids"][i], torch.Tensor):
                        pad_seq = torch.ones(longest_seqlen - encoding["input_ids"][i].shape[1], dtype=torch.long) * self._pad_id
                        pad_seq = pad_seq.unsqueeze(0) # shape: [seqlen] -> [1, seqlen]
                        pad_attn = torch.zeros(longest_seqlen - encoding["input_ids"][i].shape[1], dtype=torch.long)
                        pad_attn = pad_attn.unsqueeze(0) # shape: [seqlen] -> [1, seqlen]
                        if self.padding_side == "left":
                            encoding["input_ids"][i] = torch.cat([pad_seq, encoding["input_ids"][i]], dim=1)
                            encoding["attention_mask"][i] = torch.cat([pad_attn, encoding["attention_mask"][i]], dim=1)
                        elif self.padding_side == "right":
                            encoding["input_ids"][i] = torch.cat([encoding["input_ids"][i], pad_seq], dim=1)
                            encoding["attention_mask"][i] = torch.cat([encoding["attention_mask"][i], pad_attn], dim=1)
                    else:
                        raise ValueError(f"Unsupported type: {type(encoding['input_ids'][i])}")
            if isinstance(encoding["input_ids"][i], torch.Tensor):
                encoding["input_ids"] = torch.cat(encoding["input_ids"], dim=0)
                encoding["attention_mask"] = torch.cat(encoding["attention_mask"], dim=0)
            return encoding

    # To unified the APIs with huggingface, we add a encode function
    def encode(self, text, add_special_tokens=None):
        return self.tokenizer.encode_as_ids(text)

    def tokenize(self, text):
        return self.tokenizer.encode_as_ids(text)

    # To unified the APIs with huggingface, we changed the api from `detokenize` to `decode`
    def decode(self, ids, skip_special_tokens=False):
        if skip_special_tokens:
            assert isinstance(ids, (list, tuple)), "ids must be a list or tuple"
            ids = [id for id in ids if id not in [self._pad_id, self._bos_id, self._eos_id]]
        return self.tokenizer.decode_ids(ids)
    
    # To unified the APIs with huggingface, we add `batch_decode`
    def batch_decode(self, ids):
        # TODO: support batch decoding
        return self.tokenizer.decode_ids(ids)

    @property
    def cls(self):
        return -1

    @property
    def sep(self):
        return -1

    @property
    def mask(self):
        return -1

    # To unified the APIs with huggingface, we changed the api from `eod` to `eos_token_id`
    @property
    def eos_token_id(self):
        return self._eos_id

    @property
    def additional_special_tokens_ids(self):
        return None
    
if __name__ == '__main__':
    # Natural Questions (NQ)
    strings = ('Answer these questions:\nQ: i was a great islamic scholar and mathematician who died in 1131 ce?\nA:', 'Answer these questions:\nQ: what is the name of the main artery which takes blood from the heart to the body?\nA:', 'Answer these questions:\nQ: when was the south asian association for regional co-operation (saarc) formed?\nA:', 'Answer these questions:\nQ: what was the initial effect of the transition from command to market economies in eastern europe?\nA:', 'Answer these questions:\nQ: who captained the first european ship to sail around the tip of africa?\nA:', "Answer these questions:\nQ: how many games in a row have the uconn women's basketball team won?\nA:", 'Answer these questions:\nQ: what is the maximum data rate for the 802.11a standard select one?\nA:', 'Answer these questions:\nQ: when is the last time the vikings were in the nfc championship?\nA:', 'Answer these questions:\nQ: points on a sphere or angles in a circle are measured in units called?\nA:', 'Answer these questions:\nQ: who has the power (judicial) to make decisions in courts of law?\nA:', 'Answer these questions:\nQ: when was the last time oklahoma won a national championship in football?\nA:', 'Answer these questions:\nQ: where was percy jackson and the olympians filmed?\nA:', 'Answer these questions:\nQ: who plays captain phasma in star wars the force awakens?\nA:', 'Answer these questions:\nQ: an object that moves around an external axis is said to be?\nA:', 'Answer these questions:\nQ: what age do you need to be to buy a bb gun?\nA:', 'Answer these questions:\nQ: who does eric end up with in that 70s show?\nA:')

    tokenizer = _GPTSentencePieceTokenizer(model_file="/home/hc29225/Documents/q_mamba/mamba2-8b-converted/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model")
    truncation = False
    add_special_tokens = False
    encoding = tokenizer(
        strings,
        truncation=truncation,
        padding="longest",
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
    )
    print(encoding["input_ids"])
    print(encoding["attention_mask"])