# coding=utf-8
# Copyright 2022 EleutherAI and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes for GPTNeoXJapanese."""
import os
import json
import re
import numpy as np
import collections
from typing import TYPE_CHECKING, List

from ...tokenization_utils_fast import PreTrainedTokenizer
from ...utils import logging


if TYPE_CHECKING:
    from transformers.pipelines.conversational import Conversation


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.text", "emoji_file": "emoji.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "ABEJA/gpt-neox-japanese": "vocab.txt",
    },
    "emoji_file": {
        "ABEJA/gpt-neox-japanese": "emoji.json",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "gpt-neox-japanese": 2048,
}


def load_vocab_and_emoji(vocab_file, emoji_file):
    """Loads a vocabulary file into a dictionary."""
    with open(emoji_file, "r", encoding="utf-8") as f:
        emoji = json.loads(f.read())

    vocab = collections.OrderedDict()
    ids_to_tokens = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as f:
        token = f.read().split("\n")
    token = [[t] if (t == "," or "," not in t) else t.split(",") for t in token]
    for idx, b in enumerate(token):
        ids_to_tokens[idx] = b
        for wd in b:
            vocab[wd] = idx

    return vocab, ids_to_tokens, emoji


class GPTNeoXJapaneseTokenizer(PreTrainedTokenizer):
    """TODO: Add support
    This tokenizer inherits from [`PreTrainedTokenizer`] and based on japanese special Byte-Pair-Encoding.

    ```
    TODO: confirm sample
    >>> from transformers import GPTNeoXJapaneseokenizer
    >>> tokenizer = GPTNeoXJapaneseTokenizer.from_pretrained("ABEJA/gpt-neox-japanese")
    >>> tokenizer("吾輩は猫である")["input_ids"]
    [30014, 26883, 26638, 27228, 25, 26650]
    >>> tokenizer.decode(tokenizer("吾輩は猫である")["input_ids"])
    '吾 輩 は 猫 であ る'
    ```

    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        emoji_file (`str`):
            File containing the emoji.
        do_clean_text (`bool`, *optional*, defaults to `False`):
            Whether or not to clean text for URL, EMAIL, TEL, Japanese DATE and Japanese PRICE.
        unk_token (`str`, *optional*, defaults to `<|endoftext|>`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `<|endoftext|>`):
            The token used for padding
        bos_token (`str`, *optional*, defaults to `<|endoftext|>`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `<|endoftext|>`):
            The end of sequence token.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file=None,
        emoji_file=None,
        do_clean_text=False,
        unk_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        **kwargs
    ):
        super().__init__(
            do_clean_text=do_clean_text,
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            **kwargs,
        )
        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = GPTNeoXJapaneseokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.do_clean_text = do_clean_text
        self.vocab, self.ids_to_tokens, self.emoji = load_vocab_and_emoji(vocab_file, emoji_file)
        self.subword_tokenizer = SubWordJapaneseTokenizer(
            vocab=self.vocab, ids_to_tokens=self.ids_to_tokens, emoji=self.emoji
        )

    @property
    def vocab_size(self):
        # self.vocab contains support for character fluctuation unique to Japanese, and has a large number of vocab
        return len(self.ids_to_tokens)

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text):
        return self.subword_tokenizer.tokenize(text, clean=self.do_clean_text)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.subword_tokenizer.convert_id_to_token(index)

    def _build_conversation_input_ids(self, conversation: "Conversation") -> List[int]:
        """This corresponds to DialoGPT variants of models."""
        input_ids = []
        for is_user, text in conversation.iter_texts():
            input_ids.extend(self.encode(text, add_special_tokens=False) + [self.eos_token_id])

        if len(input_ids) > self.model_max_length:
            input_ids = input_ids[-self.model_max_length:]
        return input_ids


class SubWordJapaneseTokenizer(object):
    """
    https://github.com/tanreinama/Japanese-BPEEncoder_V2
    """

    def __init__(self, vocab, ids_to_tokens, emoji):
        self.vocab = vocab  # same as swe
        self.ids_to_tokens = ids_to_tokens  # same as bpe
        self.emoji = emoji
        self.maxlen = np.max([len(w) for w in self.vocab.keys()])
        self.content_repatter1 = re.compile(r"(https?|ftp)(:\/\/[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+$,%#]+)")
        self.content_repatter2 = re.compile(r"[A-Za-z0-9\._+]*@[\-_0-9A-Za-z]+(\.[A-Za-z]+)*")
        self.content_repatter3 = re.compile(r"[\(]{0,1}[0-9]{2,4}[\)\-\(]{0,1}[0-9]{2,4}[\)\-]{0,1}[0-9]{3,4}")
        self.content_repatter4 = re.compile(
            r"([12]\d{3}[/\-年])*(0?[1-9]|1[0-2])[/\-月]((0?[1-9]|[12][0-9]|3[01])日?)*(\d{1,2}|:|\d{1,2}時|\d{1,2}分|\(日\)|\(月\)|\(火\)|\(水\)|\(木\)|\(金\)|\(土\)|㈰|㈪|㈫|㈬|㈭|㈮|㈯)*"
        )
        self.content_repatter5 = re.compile(
            r"(明治|大正|昭和|平成|令和|㍾|㍽|㍼|㍻|\u32ff)\d{1,2}年(0?[1-9]|1[0-2])月(0?[1-9]|[12][0-9]|3[01])日(\d{1,2}|:|\d{1,2}時|\d{1,2}分|\(日\)|\(月\)|\(火\)|\(水\)|\(木\)|\(金\)|\(土\)|㈰|㈪|㈫|㈬|㈭|㈮|㈯)*"
        )
        self.content_repatter6 = re.compile(
            r"((0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*億)*((0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*万)*((0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*千)*(0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*(千円|万円|千万円|円|千ドル|万ドル|千万ドル|ドル|千ユーロ|万ユーロ|千万ユーロ|ユーロ)+(\(税込\)|\(税抜\)|\+tax)*"
        )
        keisen = "─━│┃┄┅┆┇┈┉┊┋┌┍┎┏┐┑┒┓└┕┖┗┘┙┚┛├┝┞┟┠┡┢┣┤┥┦┧┨┩┪┫┬┭┮┯┰┱┲┳┴┵┶┷┸┹┺┻┼┽┾┿╀╁╂╃╄╅╆╇╈╉╊╋╌╍╎╏═║╒╓╔╕╖╗╘╙╚╛╜╝╞╟╠╡╢╣╤╥╦╧╨╩╪╫╬╭╮╯╰╱╲╳╴╵╶╷╸╹╺╻╼╽╾╿"
        blocks = "▀▁▂▃▄▅▆▇█▉▊▋▌▍▎▏▐░▒▓▔▕▖▗▘▙▚▛▜▝▞▟"
        self.content_trans1 = str.maketrans({k: "<BLOCK>" for k in keisen + blocks})

    def __len__(self):
        return len(self.ids_to_tokens)

    def clean_text(self, content):
        content = self.content_repatter1.sub("<URL>", content)
        content = self.content_repatter2.sub("<EMAIL>", content)
        content = self.content_repatter3.sub("<TEL>", content)
        content = self.content_repatter4.sub("<DATE>", content)
        content = self.content_repatter5.sub("<DATE>", content)
        content = self.content_repatter6.sub("<PRICE>", content)
        content = content.translate(self.content_trans1)
        while "<BLOCK><BLOCK>" in content:
            content = content.replace("<BLOCK><BLOCK>", "<BLOCK>")
        return content

    def tokenize(self, text, clean=False):
        text = text.replace(" ", "<SP>")
        text = text.replace("　", "<SP>")
        text = text.replace("\r\n", "<BR>")
        text = text.replace("\n", "<BR>")
        text = text.replace("\r", "<BR>")
        text = text.replace("\t", "<TAB>")
        text = text.replace("—", "ー")
        text = text.replace("−", "ー")
        for k, v in self.emoji["emoji"].items():
            if k in text:
                text = text.replace(k, v)
        if clean:
            text = self.clean_text(text)

        def check_simbol(x):
            e = x.encode()
            if len(x) == 1 and len(e) == 2:
                c = (int(e[0]) << 8) + int(e[1])
                if (
                    (c >= 0xC2A1 and c <= 0xC2BF)
                    or (c >= 0xC780 and c <= 0xC783)
                    or (c >= 0xCAB9 and c <= 0xCBBF)
                    or (c >= 0xCC80 and c <= 0xCDA2)
                ):
                    return True
            return False

        def checku2e(x):
            e = x.encode()
            if len(x) == 1 and len(e) == 3:
                c = (int(e[0]) << 16) + (int(e[1]) << 8) + int(e[2])
                if c >= 0xE28080 and c <= 0xE2B07F:
                    return True
            return False

        pos = 0
        result = []
        while pos < len(text):
            end = min(len(text), pos + self.maxlen + 1) if text[pos] == "<" else pos + 3
            candidates = []  # (token_id, token, pos)
            for e in range(end, pos, -1):
                wd = text[pos:e]
                if wd in self.vocab:
                    if wd[0] == "<" and len(wd) > 2:
                        candidates = [(self.vocab[wd], wd, e)]
                        break
                    else:
                        candidates.append((self.vocab[wd], wd, e))
            if len(candidates) > 0:
                # the smallest token_id is adopted
                _, wd, e = sorted(candidates, key=lambda x: x[0])[0]
                result.append(wd)
                pos = e
            else:
                end = pos + 1
                wd = text[pos:end]
                if check_simbol(wd):
                    result.append("<KIGOU>")
                elif checku2e(wd):
                    result.append("<U2000U2BFF>")
                else:
                    for i in wd.encode("utf-8"):
                        result.append("<|byte%d|>" % i)
                pos = end
        return result

    def convert_id_to_token(self, index, breakline="\n"):
        words = []
        byte_tokens = []
        word = self.ids_to_tokens[index][0]
        if word[:6] == "<|byte" and word[-2:] == "|>":
            byte_tokens.append(int(word[6:-2]))
        else:
            if len(byte_tokens) > 0:
                words.append(bytearray(byte_tokens).decode("utf-8", errors="replace"))
                byte_tokens = []
            if word[:7] == "<|emoji" and word[-2:] == "|>":
                words.append(self.emoji["emoji_inv"][word])
            elif word == "<SP>":
                words.append(" ")
            elif word == "<BR>":
                words.append(breakline)
            elif word == "<TAB>":
                words.append("\t")
            elif word == "<BLOCK>":
                words.append("▀")
            elif word == "<KIGOU>":
                words.append("ǀ")
            elif word == "<U2000U2BFF>":
                words.append("‖")
            else:
                words.append(word)
        if len(byte_tokens) > 0:
            words.append(bytearray(byte_tokens).decode("utf-8", errors="replace"))
        text = "".join(words)
        return text
