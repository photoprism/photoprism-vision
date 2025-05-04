from pydantic import BaseModel


class Text(BaseModel):
    text: str


class Caption(BaseModel):
    caption: Text


class Model(BaseModel):
    name: str
    version: str


class Label(BaseModel):
    name: str
    source: str | None = None
    priority: int | None = None
    confidence: float
    topicality: float | None = None
    categories: list[str] | None = None


class Labels(BaseModel):
    labels: list[Label]


class ApiResponse(BaseModel):
    id: str
    result: Caption | Labels
    model: Model
