import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PLAYER_DETECTION_MODEL_ID = "basketball-players-fy4c2/25"
    FIELD_DETECTION_MODEL_ID = "basketball_court_segmentation-tlfja/5"
    HANDLER_DETECTION_MODEL_ID = "basketball_and_hoops/3"
    
    STRIDE = 30
    TEXT_DURATION_FRAMES = 30
    
    HANDEL = 0
    BALL_ID = 1
    HOOP_ID = 2
    PLAYER_ID = 4
    REFEREE_ID = 5
    
    @property
    def ROBOFLOW_API_KEY(self):
        return os.getenv("ROBOFLOW_API_KEY")

settings = Settings()