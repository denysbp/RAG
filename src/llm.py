from typing import cast
from torch import (
    long,
    cuda,
    float16,
    float32,
    tensor,
    Tensor,
    inference_mode
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

CACHE = ".cache/"

def get_configs() -> tuple:
    if cuda.is_available():
        device = "cuda"
        dtype = float16
    else:
        device = "cpu"
        dtype = float32
    return device, dtype

class QwenLlm:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

        self._device, dtype = get_configs()

        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=self._device
        )

        self._model.eval()

    def encode(self, prompt: str) -> Tensor:
        ids = self._tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )

        return tensor(
            [ids],
            device=self._device,
            dtype=long,
        )

    def encode_chat(
        self,
        messages: list[dict[str, str]],
        enable_thinking: bool = True,
    ) -> Tensor:
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
        )

        input_ids = cast(Tensor, inputs["input_ids"])
        return input_ids.to(self._device)

    def decode(self, ids: Tensor | list[int]) -> str:
        if isinstance(ids, Tensor):
            ids = ids.tolist()

        return cast(
            str,
            self._tokenizer.decode(
                ids,
                skip_special_tokens=True,
            ),
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 210,
        enable_thinking: bool = True,
    ) -> str:
        input_ids = self.encode_chat(
            messages,
            enable_thinking=enable_thinking,
        )

        with inference_mode():
            outputs = self._model.generate(  # type: ignore[misc]
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        new_tokens = outputs[0, input_ids.shape[1]:]
        return self.decode(new_tokens)


class SematicModel:
    def __init__(self):
        self._device, _ = get_configs()
        self._model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device=self._device,
            cache_folder=CACHE
        )

    def encode(self, sentence: str):
        embeddings = self._model.encode(
            sentence,
            device=self._device,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings