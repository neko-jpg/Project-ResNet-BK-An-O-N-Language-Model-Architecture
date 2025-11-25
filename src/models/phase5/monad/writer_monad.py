import torch
from typing import List, Optional, Any, Tuple

class WriterMonad:
    """
    Writer Monad for accumulating "Inner Speech" (Dream Core logs).

    This class acts as a buffer for the internal monologue of the agent.
    It follows the Writer Monad pattern: (Result, Log).

    Responsibilities:
    - Accumulate logs (strings or embeddings) during the inference loop.
    - Provide access to the accumulated history for the Reflector.
    - Support flushing/resetting the log buffer.
    """

    def __init__(self):
        self.log_buffer: List[str] = []
        self.embedding_buffer: List[torch.Tensor] = []

    def tell(self, message: str, embedding: Optional[torch.Tensor] = None):
        """
        Log a message (and optional embedding).

        Args:
            message: The text content of the thought.
            embedding: The vector representation of the thought (optional).
        """
        self.log_buffer.append(message)
        if embedding is not None:
            self.embedding_buffer.append(embedding)

    def listen(self) -> Tuple[List[str], Optional[torch.Tensor]]:
        """
        Return the accumulated logs.

        Returns:
            (logs, concatenated_embeddings)
        """
        if not self.embedding_buffer:
            return self.log_buffer, None

        # Concatenate embeddings along the sequence dimension (dim=0 or 1 depending on use case)
        # Assuming embeddings are (1, D) or (D,)
        try:
            concat_embed = torch.stack(self.embedding_buffer)
        except Exception:
            # Fallback if shapes don't match or other issues
            concat_embed = None

        return self.log_buffer, concat_embed

    def flush(self):
        """Clear the logs."""
        self.log_buffer = []
        self.embedding_buffer = []

    def __repr__(self):
        return f"WriterMonad(entries={len(self.log_buffer)})"
