"""
LLM client utilities for the forked bc-llm.

Backs the legacy bc-llm `LLMApi` surface (`get_output`, `get_outputs`) with
the current `lab_llm` package. Call sites in `src/` were written against
the old interface and get a drop-in replacement by importing `LLMApi` and
the Dataset classes from here.

Image inputs are not supported by this shim. The clinical-QA head-to-head
use case is text-only; re-enable images later via `lab_llm`'s content-part
support if needed.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, List, Optional

import litellm
from lab_llm import (
    CachingCompletion,
    ErrorTracker,
    LLMApi as _LabLLMApi,
    ResponseFormatError,
    wrap_completion_function,
)
from lab_llm.versa import (
    make_versa_claude_completion,
    make_versa_openai_completion,
)
from pydantic import BaseModel
from torch.utils.data import Dataset

SYSTEM_PROMPT = "You are a helpful assistant"


# ----------------------------------------------------------------------
# Vendored dataset classes
# ----------------------------------------------------------------------
# Kept here (rather than in a separate module) so `src/common.py`'s existing
# imports resolve from a single file after the llm-api-main vendor is deleted.

class TextDataset(Dataset):
    def __init__(self, prompts: List[str]):
        self.prompts = prompts

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> str:
        return self.prompts[idx]


class ImageDataset(Dataset):
    def __init__(self, image_paths, prompt_template=None, prompts=None):
        self.image_paths = image_paths
        self.prompts = prompts if prompts is not None else [prompt_template] * len(image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return self.prompts[idx], self.image_paths[idx]


class ImageGroupDataset(Dataset):
    def __init__(self, image_paths_list, prompt_template=None, prompts=None):
        self.image_paths_list = image_paths_list
        self.prompts = prompts if prompts is not None else [prompt_template] * len(image_paths_list)

    def __len__(self):
        return len(self.image_paths_list)

    def __getitem__(self, idx):
        return self.prompts[idx], "+".join(self.image_paths_list[idx])


# ----------------------------------------------------------------------
# Completion-function dispatch
# ----------------------------------------------------------------------

def _completion_for(model: str):
    if model.startswith("azure/"):
        return make_versa_openai_completion()
    if model.startswith("bedrock/"):
        return make_versa_claude_completion()
    return litellm.completion


# ----------------------------------------------------------------------
# LLMApi shim
# ----------------------------------------------------------------------

class LLMApi:
    """
    Drop-in replacement for the old bc-llm `LLMApi` surface, backed by
    `lab_llm.LLMApi`.
    """

    def __init__(
        self,
        model: str,
        cache_file: Optional[str] = None,
        seed: int = 10,
        timeout: int = 120,
        num_retries: int = 1,
        logger: Optional[logging.Logger] = None,
    ):
        self.model = model
        self.seed = seed
        self.timeout = timeout
        self._logger = logger or logging.getLogger(__name__)

        cache = CachingCompletion(cache_file) if cache_file is not None else None
        error_tracker = ErrorTracker(self._logger)

        wrapped = wrap_completion_function(
            _completion_for(model),
            cache=cache,
            error_tracker=error_tracker,
            model=model,
            seed=seed,
            timeout=timeout,
            num_retries=num_retries,
        )
        self._api = _LabLLMApi(wrapped)

    # ------------------------------------------------------------------
    # Legacy single-prompt surface
    # ------------------------------------------------------------------
    def get_output(
        self,
        prompt: str,
        max_new_tokens: int = 4000,
        temperature: float = 0,
        response_model: Optional[type[BaseModel]] = None,
    ):
        kwargs: dict[str, Any] = {
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
        if response_model is not None:
            kwargs["response_format"] = response_model
            kwargs["strict_response_format"] = True
        return self._api.run(prompt, **kwargs)

    # ------------------------------------------------------------------
    # Legacy batched surface
    # ------------------------------------------------------------------
    async def get_outputs(
        self,
        dataset: Dataset,
        batch_size: int = 10,
        max_new_tokens: int = 4000,
        is_image: bool = False,
        max_retries: int = 1,
        temperature: float = 0,
        callback=None,
        validation_func=None,
        requests_per_second=None,
        response_model: Optional[type[BaseModel]] = None,
    ) -> List[Any]:
        if is_image:
            raise NotImplementedError(
                "image inputs are not supported by the forked llm_utils shim"
            )

        prompts: List[str] = [dataset[i] for i in range(len(dataset))]
        messages_list = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": p},
            ]
            for p in prompts
        ]

        kwargs: dict[str, Any] = {
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
        if response_model is not None:
            kwargs["response_format"] = response_model
            kwargs["strict_response_format"] = True

        results = await self._api.run_batch(
            messages_list,
            max_parallel_jobs=batch_size,
            return_exceptions=True,
            **kwargs,
        )

        outputs: List[Any] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self._logger.warning("LLM call failed for prompt %d: %s", i, result)
                if isinstance(result, ResponseFormatError):
                    self._logger.warning(
                        "Full LLM output for prompt %d:\n%s", i, result.raw_content
                    )
                outputs.append(None)
            else:
                outputs.append(result)

        if callback is not None:
            callback(outputs)
        return outputs


# ----------------------------------------------------------------------
# Convenience constructor used by entry-point scripts
# ----------------------------------------------------------------------

def create_llm_clients(
    cache_file: str,
    llm_model: Optional[str] = None,
    llm_iter_model: Optional[str] = None,
    llm_extraction_model: Optional[str] = None,
    seed: int = 10,
    timeout: int = 120,
    num_retries: int = 1,
    logger: Optional[logging.Logger] = None,
) -> dict[str, LLMApi]:
    """
    Build the (iter, extraction) LLMApi pair used by `ConceptLearnerModel`.
    If both roles share the same model, a single instance is reused.
    """
    iter_model = llm_iter_model or llm_model
    extraction_model = llm_extraction_model or llm_model
    assert iter_model is not None and extraction_model is not None, (
        "must specify llm_model or both llm_iter_model and llm_extraction_model"
    )

    iter_api = LLMApi(
        iter_model,
        cache_file=cache_file,
        seed=seed,
        timeout=timeout,
        num_retries=num_retries,
        logger=logger,
    )
    if extraction_model == iter_model:
        return {"iter": iter_api, "extraction": iter_api}

    extraction_api = LLMApi(
        extraction_model,
        cache_file=cache_file,
        seed=seed,
        timeout=timeout,
        num_retries=num_retries,
        logger=logger,
    )
    return {"iter": iter_api, "extraction": extraction_api}
