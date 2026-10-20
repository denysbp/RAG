from typing import cast, Any
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
from typing import List

CACHE = ".cache/"


def get_configs() -> tuple:
    """Select the compute device and tensor type for model inference.

    Returns:
        A tuple containing the device name and tensor data type.
    """
    if cuda.is_available():
        device = "cuda"
        dtype = float16
    else:
        device = "cpu"
        dtype = float32
    return device, dtype


class QwenLlm:
    """Provide text generation through a Qwen causal language model."""

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        """Load the tokenizer and language model.

        Args:
            model_name: Hugging Face model identifier to load.
        """
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

        self._device, dtype = get_configs()

        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=self._device
        )

        self._model.eval()

    def encode(self, prompt: str) -> Tensor:
        """Tokenize a prompt into model input IDs.

        Args:
            prompt: Text prompt to tokenize.

        Returns:
            A tensor containing the token IDs on the configured device.
        """
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
        """Format and tokenize chat messages for generation.

        Args:
            messages: Chat messages with roles and content.
            enable_thinking: Whether the model's thinking mode is enabled.

        Returns:
            A tensor containing the chat input IDs.
        """
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
        """Decode token IDs into readable text.

        Args:
            ids: Token IDs represented as a tensor or list.

        Returns:
            Decoded text without special tokens.
        """
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
        """Generate a response for a chat conversation.

        Args:
            messages: Chat messages used as the generation prompt.
            max_new_tokens: Maximum number of tokens to generate.
            enable_thinking: Whether the model's thinking mode is enabled.

        Returns:
            The generated response text.
        """
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
    """Provide normalized sentence embeddings for semantic retrieval."""

    def __init__(self) -> None:
        """Load the sentence-transformer embedding model."""
        self._device, _ = get_configs()
        self._model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device=self._device,
            cache_folder=CACHE
        )

    def encode(
        self,
        sentence: List[str],
        show: bool = False
    ) -> Any:
        """Encode sentences as normalized NumPy embeddings.

        Args:
            sentence: Sentences to encode.
            show: Whether to display the model progress bar.

        Returns:
            The generated sentence embeddings.
        """
        embeddings = self._model.encode(
            sentence,
            device=self._device,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show
        )
        return embeddings
