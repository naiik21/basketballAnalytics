import supervision as sv
from config.settings import settings
from config.constants import Colors

def setup_video_sources(source_path, target_path, stride=None):
    video_info = sv.VideoInfo.from_video_path(source_path)
    video_sink = sv.VideoSink(target_path, video_info=video_info)
    
    if stride:
        frame_generator = sv.get_video_frames_generator(
            source_path=source_path, 
            stride=stride
        )
    else:
        frame_generator = sv.get_video_frames_generator(source_path)
    
    return video_info, video_sink, frame_generator

def collect_player_crops(frame_generator, model, device=settings.device):
    crops = []
    for frame in frame_generator:
        result = model.infer(frame, device=device, confidence=0.3)[0]
        detections = sv.Detections.from_inference(result)
        detections = detections.with_nms(threshold=0.5, class_agnostic=True)
        detections = detections[detections.class_id == settings.PLAYER_ID]
        players_crops = [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]
        crops += players_crops
    return crops