import os
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from docling_core.types.doc.page import SegmentedPage
from pydantic import AnyUrl, BaseModel, model_validator
from typing_extensions import deprecated

from docling.datamodel.accelerator_options import AcceleratorDevice


class BaseVlmOptions(BaseModel):
    kind: str
    prompt: Union[str, Callable[[Optional[SegmentedPage]], str]]
    scale: float = 2.0
    max_size: Optional[int] = None
    temperature: float = 0.0


class ResponseFormat(str, Enum):
    DOCTAGS = "doctags"
    MARKDOWN = "markdown"
    HTML = "html"


class InferenceFramework(str, Enum):
    MLX = "mlx"
    TRANSFORMERS = "transformers"


class TransformersModelType(str, Enum):
    AUTOMODEL = "automodel"
    AUTOMODEL_VISION2SEQ = "automodel-vision2seq"
    AUTOMODEL_CAUSALLM = "automodel-causallm"
    AUTOMODEL_IMAGETEXTTOTEXT = "automodel-imagetexttotext"


class TransformersPromptStyle(str, Enum):
    CHAT = "chat"
    RAW = "raw"


class InlineVlmOptions(BaseVlmOptions):
    kind: Literal["inline_model_options"] = "inline_model_options"

    repo_id: str
    trust_remote_code: bool = False
    load_in_8bit: bool = True
    llm_int8_threshold: float = 6.0
    quantized: bool = False

    inference_framework: InferenceFramework
    transformers_model_type: TransformersModelType = TransformersModelType.AUTOMODEL
    transformers_prompt_style: TransformersPromptStyle = TransformersPromptStyle.CHAT
    response_format: ResponseFormat

    torch_dtype: Optional[str] = None
    supported_devices: List[AcceleratorDevice] = [
        AcceleratorDevice.CPU,
        AcceleratorDevice.CUDA,
        AcceleratorDevice.MPS,
    ]

    stop_strings: List[str] = []
    extra_generation_config: Dict[str, Any] = {}

    use_kv_cache: bool = True
    max_new_tokens: int = 4096

    @model_validator(mode="before")
    @classmethod
    def check_vlm_quantization_env(cls, data: Any) -> Any:
        """Check for VLM quantization environment variables."""
        if isinstance(data, dict):
            # Check if quantization should be enabled via env var
            vlm_quantize = os.getenv("DOCLING_VLM_QUANTIZE_8BIT")
            if vlm_quantize is not None and data.get("quantized") is None:
                # Convert string to bool (accepts: true, 1, yes, on)
                data["quantized"] = vlm_quantize.lower() in {"true", "1", "yes", "on"}
                if data["quantized"]:
                    # Also set load_in_8bit if not explicitly set
                    if data.get("load_in_8bit") is None:
                        data["load_in_8bit"] = True
        return data

    @property
    def repo_cache_folder(self) -> str:
        return self.repo_id.replace("/", "--")


@deprecated("Use InlineVlmOptions instead.")
class HuggingFaceVlmOptions(InlineVlmOptions):
    pass


class ApiVlmOptions(BaseVlmOptions):
    kind: Literal["api_model_options"] = "api_model_options"

    url: AnyUrl = AnyUrl(
        "http://localhost:11434/v1/chat/completions"
    )  # Default to ollama
    headers: Dict[str, str] = {}
    params: Dict[str, Any] = {}
    timeout: float = 60
    concurrency: int = 1
    response_format: ResponseFormat
