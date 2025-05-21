from inference import get_model
from config.settings import settings

class DetectionModel:
    def __init__(self, model_id, api_key=settings.ROBOFLOW_API_KEY):
        self.model = get_model(model_id=model_id, api_key=api_key)
    
    def infer(self, frame, confidence=0.3, device="cuda"):
        return self.model.infer(frame, confidence=confidence, device=device)

class PlayerDetectionModel(DetectionModel):
    def __init__(self):
        super().__init__(settings.PLAYER_DETECTION_MODEL_ID)

class FieldDetectionModel(DetectionModel):
    def __init__(self):
        super().__init__(settings.FIELD_DETECTION_MODEL_ID)

class HandlerDetectionModel(DetectionModel):
    def __init__(self):
        super().__init__(settings.HANDLER_DETECTION_MODEL_ID)