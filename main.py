import os
import time
import torch
from tqdm import tqdm
import supervision as sv

# Configuración
from config.settings import settings

# Modelos
from models.detection import PlayerDetectionModel, FieldDetectionModel, HandlerDetectionModel
from models.teamClassifier import TeamClassifier

# Procesamiento
from processing.tracking import BasketballTracker
from processing.annotations import Annotator
from processing.zone_analysis import is_inside, analyze_shooting_zone

# Utilidades
from utils.visualization import setup_video_sources, collect_player_crops

def main():
    # Inicialización de modelos
    player_model = PlayerDetectionModel()
    field_model = FieldDetectionModel()
    handler_model = HandlerDetectionModel()
    tracker = BasketballTracker()
    annotator = Annotator()
    
    # Configuración de video
    SOURCE_VIDEO_PATH = "./videos/test/stepback.mp4"
    TARGET_VIDEO_PATH = "./videos/result/shastepback_result_3.mp4"
    
    # Recolección de crops para clasificación de equipos
    print("Collecting player crops for team classification...")
    frame_generator = setup_video_sources(SOURCE_VIDEO_PATH, None, settings.STRIDE)[2]
    crops = collect_player_crops(frame_generator, player_model)
    
    # Entrenamiento del clasificador de equipos
    print("Training team classifier...")
    team_classifier = TeamClassifier(device="cuda")
    team_classifier.fit(crops)
    
    # Procesamiento del video principal
    print("Processing video...")
    video_info, video_sink, frame_generator = setup_video_sources(
        SOURCE_VIDEO_PATH, TARGET_VIDEO_PATH
    )
    
    # Variables de estado para el texto NBA
    active_text = None
    remaining_frames = 0
    ball_in_hoop = False
    
    with video_sink:
        for frame in tqdm(frame_generator, total=video_info.total_frames):
            # Detección y seguimiento de objetos
            detections = tracker.process_frame(frame, player_model)
            
            # Clasificación de equipos para jugadores
            players_crops = [sv.crop_image(frame, xyxy) for xyxy in detections['players'].xyxy]
            detections['players'].class_id = team_classifier.predict(players_crops)
            
            # Detección del manejador (jugador con balón)
            handler_result = handler_model.infer(frame, confidence=0.3)[0]
            handler_detections = sv.Detections.from_inference(handler_result)
            handler_detections = handler_detections[handler_detections.class_id == settings.HANDEL]
            
            if len(handler_detections) > 0:
                max_confidence_idx = int(handler_detections.confidence.argmax())
                handler_detections = handler_detections[max_confidence_idx]
            
            # Verificar si el balón está en el aro
            try:
                ball_in_hoop = is_inside(
                    detections['ball'].xyxy[0], 
                    detections['hoop'].xyxy[0], 
                    partial_overlap=False
                )
            except:
                pass
            
            # Análisis de zona de tiro
            field_result = field_model.infer(frame, confidence=0.3)[0]
            zones = sv.Detections.from_inference(field_result)
            
            shot_type = analyze_shooting_zone(
                frame, zones, handler_detections, detections['ball']
            )
            
            # Manejo del texto NBA
            if remaining_frames > 0:
                remaining_frames -= 1
                annotator.draw_nba_style_text(annotated_frame, active_text)
                if remaining_frames == 0:
                    active_text = None
                    ball_in_hoop = False
            elif ball_in_hoop and shot_type:
                active_text = shot_type
                remaining_frames = settings.TEXT_DURATION_FRAMES
                ball_in_hoop = False
            
            # Anotación del frame
            annotated_frame = frame.copy()
            annotated_frame = annotator.annotate_frame(
                annotated_frame, 
                detections, 
                handler_detections, 
                zones, 
                ball_in_hoop
            )
            
            video_sink.write_frame(annotated_frame)

if __name__ == "__main__":
    main()