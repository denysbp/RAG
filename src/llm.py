from typing import cast
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenLlm:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

        if torch.cuda.is_available():
            self._device = "cuda"
            dtype = torch.float16
        else:
            self._device = "cpu"
            dtype = torch.float32

        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=self._device
        )

        self._model.eval()

    def encode(self, prompt: str) -> torch.Tensor:
        ids = self._tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )

        return torch.tensor(
            [ids],
            device=self._device,
            dtype=torch.long,
        )

    def encode_chat(
        self,
        messages: list[dict[str, str]],
        enable_thinking: bool = True,
    ) -> torch.Tensor:
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

        input_ids = cast(torch.Tensor, inputs["input_ids"])
        return input_ids.to(self._device)

    def decode(self, ids: torch.Tensor | list[int]) -> str:
        if isinstance(ids, torch.Tensor):
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
        max_new_tokens: int = 200,
        enable_thinking: bool = True,
    ) -> str:
        input_ids = self.encode_chat(
            messages,
            enable_thinking=enable_thinking,
        )

        with torch.inference_mode():
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