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


class NSFWProbabilities(BaseModel):
    Neutral: float  # >0.25 means the image is not NSFW
    Drawing: float  # value between 0 and 1, higher value means it is more likely to be a drawing
    Hentai: float  # value between 0 and 1, higher value means it is more likely to be a hentai
    Porn: float  # value between 0 and 1, higher value means it is more likely to be porn
    Sexy: float  # value between 0 and 1, higher value means it is more likely to be a sexy content


class NSFW(BaseModel):
    nsfw: list[NSFWProbabilities]


class ApiResponse(BaseModel):
    id: str
    result: Caption | Labels | NSFW
    model: Model
