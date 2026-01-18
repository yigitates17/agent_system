# infra/response_validator.py

import json
from pydantic import BaseModel, ValidationError
from core.schemas import LLMResponse


class ValidationResult(BaseModel):
    valid: bool
    error: str | None = None
    retry_prompt: str | None = None


class ResponseValidator:
    def validate_json(self, response: LLMResponse) -> ValidationResult:
        if not response.content:
            return ValidationResult(
                valid=False,
                error="Empty response",
                retry_prompt="You returned an empty response. Please try again.",
            )
        
        try:
            json.loads(response.content)
            return ValidationResult(valid=True)
        except json.JSONDecodeError as e:
            return ValidationResult(
                valid=False,
                error=str(e),
                retry_prompt=f"Your response was not valid JSON: {e}. Return only valid JSON.",
            )
    
    def validate_with_model(
        self, 
        response: LLMResponse, 
        model: type[BaseModel],
    ) -> ValidationResult:
        if not response.content:
            return ValidationResult(
                valid=False,
                error="Empty response",
                retry_prompt="You returned an empty response. Please try again.",
            )
        
        try:
            model.model_validate_json(response.content)
            return ValidationResult(valid=True)
        except ValidationError as e:
            return ValidationResult(
                valid=False,
                error=str(e),
                retry_prompt=f"Your response didn't match the expected format: {e}. Fix it.",
            )