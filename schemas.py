from pydantic import BaseModel, Field

class VaccineResponse(BaseModel):
    likelihood: int = Field(..., ge=1, le=5, description="Likelihood of taking the vaccine (1-5)")
    confidence: int = Field(..., ge=1, le=5, description="Confidence in the response (1-5)")
    emotion: str = Field(..., description="A single sentence describing emotional reaction to the article")