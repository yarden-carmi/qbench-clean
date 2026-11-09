"""Model providers for Hugging Face (transformers, timm) models with small real datasets.

``hf_provider(spec)`` returns a :class:`qbench.providers.ModelProvider` and the
qbench evaluation task to use with it. ``spec`` keys: ``id``, ``kind`` (hf | timm),
``task`` (causal-lm, masked-lm, feature, seq2seq, speech-seq2seq, ctc, clip,
image-classification...), optional ``repo``, ``samples``, ``seq_len``.

Datasets (downloaded once into the Hugging Face cache):

- text: wikitext-2-raw-v1, test split
- images: timm/mini-imagenet (validation, 100 ImageNet classes, shuffled), labels mapped to
  ImageNet-1k ids through qbench/data/imagenet_class_index.json
- speech: hf-internal-testing/librispeech_asr_dummy

When a dataset cannot be loaded the provider falls back to synthetic inputs and
says so in ``provider.dataset`` (agreement metrics stay meaningful; task
accuracy does not).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from qbench.schemas import Scenario

def _imagenet_rows(samples: int):
    """(PIL image, ImageNet-1k label) pairs from timm/mini-imagenet's validation split."""
    import json
    from importlib.resources import files

    from datasets import load_dataset
    index = json.loads(files("qbench.data").joinpath("imagenet_class_index.json").read_text())
    wnid_to_idx = {v[0]: int(k) for k, v in index.items()}
    ds = load_dataset("timm/mini-imagenet", split="validation", streaming=True).shuffle(seed=0, buffer_size=2000)
    names = ds.features["label"].names
    for i, row in enumerate(ds):
        if i >= samples:
            break
        yield row["image"].convert("RGB"), wnid_to_idx[names[row["label"]]]


@dataclass
class HFProvider:
    spec: dict[str, Any]
    samples: int = 16
    seq_len: int = 256
    dataset: str = ""
    _batches: list[Any] = field(default_factory=list)
    _model: nn.Module | None = None

    # ------------------------------------------------------------------ model
    def build_model(self) -> nn.Module:
        raise NotImplementedError

    def clone_model(self, model: nn.Module) -> nn.Module:
        return copy.deepcopy(model)

    def capture_scenarios(self):
        return [self.prepare_evaluation_batch(self.evaluation_loader()[0])[0]]

    def select_metric_output(self, output: Any) -> Any:
        for key in ("logits", "logits_per_image", "last_hidden_state", "pred_masks"):
            if hasattr(output, key) and getattr(output, key) is not None:
                return getattr(output, key)
            if isinstance(output, dict) and key in output:
                return output[key]
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    # ------------------------------------------------------------------ data
    def evaluation_loader(self) -> list[Any]:
        if not self._batches:
            self._batches = self._load()
        return self._batches

    def prepare_evaluation_batch(self, batch: Any):
        kwargs, target = batch
        scenario = Scenario("evaluation", (), kwargs)
        return (scenario, target) if target is not None else scenario

    def _load(self) -> list[Any]:
        raise NotImplementedError


def _wikitext_text() -> str:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return "\n".join(t for t in ds["text"] if t.strip())


class CausalLMProvider(HFProvider):
    def build_model(self):
        if self._model is None:
            from transformers import AutoModelForCausalLM
            repo = self.spec.get("repo") or self.spec["id"]
            self._model = AutoModelForCausalLM.from_pretrained(
                repo, torch_dtype=torch.float32, attn_implementation="eager").eval()
        return self._model

    def _load(self):
        from transformers import AutoTokenizer
        repo = self.spec.get("repo") or self.spec["id"]
        tok = AutoTokenizer.from_pretrained(repo)
        try:
            ids = tok(_wikitext_text()[: 400_000], return_tensors="pt").input_ids[0]
            self.dataset = "wikitext-2-raw-v1/test"
        except Exception:  # noqa: BLE001
            ids = torch.randint(0, tok.vocab_size, (self.samples * self.seq_len,), generator=torch.Generator().manual_seed(0))
            self.dataset = "synthetic-tokens"
        out = []
        for i in range(self.samples):
            chunk = ids[i * self.seq_len:(i + 1) * self.seq_len]
            if chunk.numel() < self.seq_len:
                break
            x = chunk[None]
            labels = torch.cat([x[:, 1:], torch.full((1, 1), -100)], 1)
            out.append(({"input_ids": x, "use_cache": False}, labels))
        return out

    task = "language_modeling"


class TextEncoderProvider(HFProvider):
    task = "generic"

    def build_model(self):
        if self._model is None:
            import transformers as tf
            repo = self.spec.get("repo") or self.spec["id"]
            cls = tf.AutoModelForMaskedLM if self.spec.get("task") == "masked-lm" else tf.AutoModel
            if self.spec.get("task") == "seq2seq":
                cls = tf.AutoModelForSeq2SeqLM
            self._model = cls.from_pretrained(repo, torch_dtype=torch.float32, attn_implementation="eager").eval()
        return self._model

    def _load(self):
        from transformers import AutoTokenizer
        repo = self.spec.get("repo") or self.spec["id"]
        tok = AutoTokenizer.from_pretrained(repo)
        try:
            lines = [ln for ln in _wikitext_text().split("\n") if len(ln) > 200][: self.samples]
            self.dataset = "wikitext-2-raw-v1/test"
        except Exception:  # noqa: BLE001
            lines = ["The quick brown fox jumps over the lazy dog. " * 8] * self.samples
            self.dataset = "synthetic-text"
        out = []
        for ln in lines:
            enc = tok(ln, return_tensors="pt", truncation=True, max_length=128)
            kw = {"input_ids": enc.input_ids, "attention_mask": enc.attention_mask}
            if self.spec.get("task") == "seq2seq":
                kw["decoder_input_ids"] = enc.input_ids[:, :32]
            out.append((kw, None))
        return out


class TimmProvider(HFProvider):
    task = "classification"

    def build_model(self):
        if self._model is None:
            import timm
            self._model = timm.create_model(self.spec["id"], pretrained=True).float().eval()
        return self._model

    def _load(self):
        import timm
        model = self.build_model()
        cfg = timm.data.resolve_data_config({}, model=model)
        if self.spec.get("image_size"):
            cfg["input_size"] = (3, self.spec["image_size"], self.spec["image_size"])
        tf = timm.data.create_transform(**cfg)
        out = []
        try:
            for image, label in _imagenet_rows(self.samples):
                out.append(({"x": tf(image)[None]}, torch.tensor([label])))
            self.dataset = "timm/mini-imagenet/validation"
        except Exception:  # noqa: BLE001
            g = torch.Generator().manual_seed(0)
            out = [({"x": torch.randn(1, *cfg["input_size"], generator=g)}, None) for _ in range(self.samples)]
            self.dataset = "synthetic-images"
            self.task = "generic"
        return out

    def prepare_evaluation_batch(self, batch):
        kwargs, target = batch
        scenario = Scenario("evaluation", (kwargs["x"],), {})
        return (scenario, target) if target is not None else scenario


class ClipProvider(HFProvider):
    task = "generic"

    def build_model(self):
        if self._model is None:
            from transformers import AutoModel
            self._model = AutoModel.from_pretrained(self.spec["id"], torch_dtype=torch.float32,
                                                    attn_implementation="eager").eval()
        return self._model

    def _load(self):
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(self.spec["id"])
        texts = [f"a photo of a {c}" for c in ("tench", "dog", "cassette player", "chain saw", "church",
                                               "french horn", "garbage truck", "gas pump", "golf ball", "parachute")]
        out = []
        try:
            for image, _label in _imagenet_rows(self.samples):
                enc = proc(text=texts, images=image, return_tensors="pt", padding=True)
                out.append((dict(enc), None))
            self.dataset = "timm/mini-imagenet/validation"
        except Exception:  # noqa: BLE001
            self.dataset = "unavailable"
        return out


class SpeechProvider(HFProvider):
    task = "generic"

    def build_model(self):
        if self._model is None:
            import transformers as tf
            cls = tf.AutoModelForSpeechSeq2Seq if self.spec.get("task") == "speech-seq2seq" else tf.AutoModelForCTC
            self._model = cls.from_pretrained(self.spec["id"], torch_dtype=torch.float32,
                                              attn_implementation="eager").eval()
        return self._model

    def _load(self):
        from datasets import load_dataset
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(self.spec["id"])
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        self.dataset = "librispeech_asr_dummy/validation"
        out = []
        for row in list(ds)[: self.samples]:
            audio = row["audio"]
            enc = proc(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt")
            if self.spec.get("task") == "speech-seq2seq":
                labels = proc.tokenizer(row["text"].lower(), return_tensors="pt").input_ids
                kw = {"input_features": enc.input_features, "decoder_input_ids": labels[:, :-1]}
                self.task = "language_modeling"
                out.append((kw, labels[:, 1:]))
            else:
                out.append(({"input_values": enc.input_values}, None))
        return out


def hf_provider(spec: dict[str, Any], samples: int = 16, seq_len: int = 256) -> HFProvider:
    task = spec.get("task", "")
    if spec["kind"] == "timm":
        cls: type[HFProvider] = TimmProvider
    elif task == "causal-lm":
        cls = CausalLMProvider
    elif task in ("masked-lm", "feature", "seq2seq"):
        cls = TextEncoderProvider
    elif task == "clip":
        cls = ClipProvider
    elif task in ("speech-seq2seq", "ctc"):
        cls = SpeechProvider
    else:
        raise ValueError(f"no provider for {spec['id']} ({spec['kind']}, {task})")
    return cls(spec=spec, samples=samples, seq_len=seq_len)
