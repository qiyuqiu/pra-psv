"""Frozen final-token encoding; requires an explicitly supplied model path."""
import torch

class FrozenPrefixEncoder:
    """Extract final-token hidden states from exactly the visible cumulative prefix."""

    def __init__(self, model_path: str, *, device_map: str = "auto", max_length: int = 512,
                 trust_remote_code: bool = False):
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("install transformers>=4.40 and accelerate>=0.28") from exc
        self.model_path = model_path
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        # Qwen ships a pad token, so this went unnoticed until a non-Qwen encoder was
        # added: Mistral and Falcon have none, and BatchEncoding refuses to pad
        # without one. Padding positions are masked out of the read anyway -- encode()
        # takes the last attended token -- so reusing EOS is safe here.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.backbone = AutoModel.from_pretrained(
            model_path, device_map=device_map, torch_dtype="auto",
            trust_remote_code=trust_remote_code,
        ).eval()
        self.backbone.requires_grad_(False)
        self.hidden_size = self.backbone.config.hidden_size
        self._hidden_output, self._needs_hidden_states = self._probe_hidden_output()

    def _probe_hidden_output(self) -> tuple[str, bool]:
        """Find where this backbone exposes its final hidden states.

        Not every architecture answers `AutoModel` with a bare encoder. ChatGLM's
        remote code maps it to a causal LM, whose output carries `hidden_states`
        (and only when asked) instead of `last_hidden_state`. Probing once here
        keeps the hot encode loop free of a try/except retry per batch.
        """
        device = self.backbone.get_input_embeddings().weight.device
        tokens = self.tokenizer("probe", return_tensors="pt").to(device)
        with torch.inference_mode():
            output = self.backbone(**tokens)
            if hasattr(output, "last_hidden_state"):
                return "last_hidden_state", False
            if getattr(output, "hidden_states", None) is not None:
                return "hidden_states", False
            output = self.backbone(**tokens, output_hidden_states=True)
            if getattr(output, "hidden_states", None) is not None:
                return "hidden_states", True
        raise RuntimeError(
            f"{self.model_path}: backbone exposes neither last_hidden_state nor "
            f"hidden_states, so prefix features cannot be read from it"
        )

    @torch.inference_mode()
    def encode(self, texts: list[str], batch_size: int = 4) -> torch.Tensor:
        outputs = []
        device = self.backbone.get_input_embeddings().weight.device
        for start in range(0, len(texts), batch_size):
            tokens = self.tokenizer(texts[start:start + batch_size], padding=True,
                                    truncation=True, max_length=self.max_length,
                                    return_tensors="pt").to(device)
            if self._needs_hidden_states:
                output = self.backbone(**tokens, output_hidden_states=True)
            else:
                output = self.backbone(**tokens)
            hidden = getattr(output, self._hidden_output)
            if self._hidden_output == "hidden_states":
                hidden = hidden[-1]  # last layer, matching last_hidden_state
                # ChatGLM's release emits seq-major activations -- (seq, batch,
                # hidden) rather than the (batch, seq, hidden) every standard
                # backbone uses. Normalise against the lengths we already know
                # rather than inferring the layout from the shape, which would be
                # ambiguous whenever batch and sequence happen to be equal.
                batch_size_here = int(tokens["input_ids"].shape[0])
                seq_len_here = int(tokens["input_ids"].shape[1])
                if hidden.shape[0] == seq_len_here and hidden.shape[1] == batch_size_here:
                    hidden = hidden.transpose(0, 1)
            last = tokens["attention_mask"].sum(dim=1) - 1
            # Tokenizers may left-pad; the last attended token is then the final column.
            if self.tokenizer.padding_side == "left":
                last = torch.full_like(last, hidden.shape[1] - 1)
            outputs.append(hidden[torch.arange(hidden.shape[0], device=hidden.device), last].float().cpu())
        return torch.cat(outputs) if outputs else torch.empty(0, self.hidden_size)
