"""Response models; they also define the service's OpenAPI documentation."""

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    code: str = Field(examples=["unsupported_format"])
    message: str
    details: list[dict] | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])


class ReadyResponse(BaseModel):
    status: str = Field(examples=["ready"])
    model_version: str | None = None


class ClassProbability(BaseModel):
    label: str
    probability: float


class PredictionResult(BaseModel):
    label: str = Field(description="Predicted class")
    confidence: float = Field(description="Probability of the predicted class")
    top_k: list[ClassProbability] = Field(description="Most likely classes, highest first")
    probabilities: dict[str, float] = Field(description="Probability for every class")


class PredictionResponse(PredictionResult):
    model_version: str
    inference_ms: float = Field(description="Preprocessing and model time for this request")


class BatchItem(PredictionResult):
    index: int = Field(description="Position of the file in the request")


class BatchPredictionResponse(BaseModel):
    predictions: list[BatchItem] = Field(description="One result per file, in request order")
    model_version: str
    inference_ms: float


class SplitMetrics(BaseModel):
    images: int | None = None
    accuracy: float | None = None
    macro_f1: float | None = None
    macro_precision: float | None = None
    macro_recall: float | None = None


class ModelInfoResponse(BaseModel):
    model_version: str
    backbone: str
    class_names: list[str]
    num_classes: int
    image_size: int
    epoch: int | None
    checkpoint_md5: str
    training_commit: str
    training_tree_dirty: str
    mlflow_run_id: str | None
    device: str
    validation_metrics: dict[str, float]
    evaluation_metrics: dict[str, SplitMetrics] | None = Field(
        description="Held-out test metrics; null unless reports/evaluation matches this model"
    )
