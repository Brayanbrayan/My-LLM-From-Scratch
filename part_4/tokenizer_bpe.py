from __future__ import annotations
import os, json
from pathlib import Path
from typing import List, Union

try:
    from tokenizers import ByteLevelBPETokenizer, Tokenizer
except Exception:
    ByteLevelBPETokenizer = None


class BPETokenizer:
    """Minimal BPE wrapper (HuggingFace tokenizers).
    Trains on a text file or a folder of .txt files. Saves merges/vocab to out_dir.
    """
    def __init__(self, vocab_size: int = 32000, special_tokens: List[str] | None = None):
        if ByteLevelBPETokenizer is None:
            raise ImportError("Please 'pip install tokenizers' for BPETokenizer.")
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or ["<s>", "</s>", "<pad>", "<unk>", "<mask>"]
        self._tok = None

    def train(self, data_path: Union[str, Path]):
        files: List[str] = []
        p = Path(data_path)
        # ✅ DEBUG: Print what we're looking for
        print(f"[DEBUG] Looking for data at: {p}")
        print(f"[DEBUG] Path exists: {p.exists()}")
        print(f"[DEBUG] Is directory: {p.is_dir()}")
    
        if p.is_dir():
            files = [str(fp) for fp in p.glob("**/*.txt")]
            print(f"[DEBUG] Found {len(files)} .txt files in directory")
        else:
            files = [str(p)]
            print(f"[DEBUG] Single file: {files[0]}")
    
        # ✅ Print the exact files being passed to tokenizer
        print(f"[DEBUG] Files to train on: {files}")
    
        # ✅ Check each file exists
        for f in files:
            if not Path(f).exists():
                raise FileNotFoundError(f"File does not exist: {f}")
        tok = ByteLevelBPETokenizer()
        tok.train(files=files, vocab_size=self.vocab_size, min_frequency=2, special_tokens=self.special_tokens)
        self._tok = tok

    def save(self, out_dir: Union[str, Path]):
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        assert self._tok is not None, "Train or load before save()."
        self._tok.save(str(out / "tokenizer.json"))
        meta = {"vocab_size": self.vocab_size, "special_tokens": self.special_tokens}
        (out/"bpe_meta.json").write_text(json.dumps(meta))

    def load(self, dir_path: Union[str, Path]):
        dirp = Path(dir_path)
        tokenizer_file = dirp / "tokenizer.json"
    
        # Check if tokenizer.json exists (new format)
        if tokenizer_file.exists():
            tok = Tokenizer.from_file(str(tokenizer_file))
            self._tok = tok
        else:
            # Fallback: try old format with vocab.json and merges.txt
            vocab = dirp / "vocab.json"
            merges = dirp / "merges.txt"
        
            if not vocab.exists() or not merges.exists():
                # Last resort: glob for any json/txt files
                vs = list(dirp.glob("*.json"))
                ms = list(dirp.glob("*.txt"))
                if not vs or not ms:
                    raise FileNotFoundError(
                        f"Could not find tokenizer.json or vocab.json/merges.txt in {dirp}"
                    )
                vocab = vs[0]
                merges = ms[0]
        
            # Load using the old format
            tok = ByteLevelBPETokenizer(str(vocab), str(merges))
            self._tok = tok
    
        # Load metadata if it exists
        meta_file = dirp / "bpe_meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text())  # Fixed: was missing ()
            self.vocab_size = meta.get("vocab_size", self.vocab_size)
            self.special_tokens = meta.get("special_tokens", self.special_tokens)

    def encode(self, text: str):
        ids = self._tok.encode(text).ids
        return ids
    def decode(self, ids):
        return self._tok.decode(ids)


    
