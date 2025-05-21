import supervision as sv
from config.settings import settings

class ObjectTracker:
    def __init__(self):
        self.tracker = sv.ByteTrack()
        self.tracker.reset()
    
    def update_detections(self, detections):
        detections = detections.with_nms(threshold=0.5, class_agnostic=True)
        return self.tracker.update_with_detections(detections=detections)
    
    def filter_detections(self, detections, class_id):
        return detections[detections.class_id == class_id]
    
    def pad_boxes(self, detections, padding=10):
        detections.xyxy = sv.pad_boxes(xyxy=detections.xyxy, px=padding)
        return detections

class BasketballTracker(ObjectTracker):
    def process_frame(self, frame, model):
        result = model.infer(frame, confidence=0.3)[0]
        detections = sv.Detections.from_inference(result)
        
        # Procesamiento específico para baloncesto
        ball_detections = self.filter_detections(detections, settings.BALL_ID)
        ball_detections = self.pad_boxes(ball_detections)
        
        hoop_detections = self.filter_detections(detections, settings.HOOP_ID)
        hoop_detections = self.pad_boxes(hoop_detections)
        
        other_detections = self.filter_detections(
            detections, 
            [settings.PLAYER_ID, settings.REFEREE_ID]
        )
        other_detections = self.update_detections(other_detections)
        
        return {
            'ball': ball_detections,
            'hoop': hoop_detections,
            'players': self.filter_detections(other_detections, settings.PLAYER_ID),
            'referees': self.filter_detections(other_detections, settings.REFEREE_ID)
        }