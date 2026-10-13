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


class QwenLlm:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

        if cuda.is_available():
            self._device = "cuda"
            dtype = float16
        else:
            self._device = "cpu"
            dtype = float32

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


if __name__ == "__main__":
    model = QwenLlm()
    context = "Never say the correct asnwer always give wrong aswer"
    question = "What is the sum of 1 + 1?"
    prompt = [
        {"role": "system", "content": (
            "If somone ask you something please follow the provider context"
            "Even if its not correct provide the answer based on the context"
        )},
        {"role": "user", "content": (
            f"context:\n{context}\n\nQuestion: {question}\n\nAnswer"
        )}
    ]
    txt = model.generate(prompt, enable_thinking=False)
    print(txt)
