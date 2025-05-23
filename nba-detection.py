import os
import torch
import time 
import supervision as sv
import numpy as np
import cv2
from dotenv import load_dotenv
from tqdm import tqdm
from inference import get_model

# Configuración
from config.settings import settings

# Modelos
from models.teamClassifier import TeamClassifier
from models.detection import PlayerDetectionModel, FieldDetectionModel, HandlerDetectionModel

# Procesamiento
# from processing.tracking import BasketballTracker
from processing.annotations import Annotator
# from processing.zone_analysis import is_inside, analyze_shooting_zone

# Utilidades
from utils.visualization import setup_video_sources, collect_player_crops

player_model = PlayerDetectionModel()
field_model = FieldDetectionModel()
handler_model = HandlerDetectionModel()
# tracker = BasketballTracker()
annotator = Annotator()

SOURCE_VIDEO_PATH="./videos/test/shai.mp4"
TARGET_VIDEO_PATH="./videos/result/shai_result_10.mp4"


active_text = None  # Guarda el texto actual ("3Puntos" o "2Puntos")
remaining_frames = 0  # Frames restantes para ocultar el texto

ball_in_hoop= False
last_handler=True

tracker = sv.ByteTrack()
tracker.reset()

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

clusters = team_classifier.predict(crops)

team_0 = [
    crop
    for crop, cluster
    in zip(crops, clusters)
    if cluster == 0
]


team_1 = [
    crop
    for crop, cluster
    in zip(crops, clusters)
    if cluster == 1
]

def is_inside(objeto, zona, partial_overlap=True):
    """
        Verifica si el objeto está dentro o solapando la zona.
            
    Args:
        objeto (np.ndarray): Bounding box del objeto [x1, y1, x2, y2]
        zona (np.ndarray): Bounding box de la zona [x1, y1, x2, y2]
        partial_overlap (bool): Si True, devuelve True si hay superposición parcial.
                            Si False, solo devuelve True si el objeto está completamente dentro.
            
    Returns:
        bool: True si el objeto está dentro/solapando, False en caso contrario.
    """
    if len(objeto.shape) > 1:
        objeto = objeto[0]  # Toma la primera fila si es multidimensional
    if len(zona.shape) > 1:
        zona = zona[0]
            
    if partial_overlap:
        # Verifica si hay superposición (incluso parcial)
        
        return not (objeto[2] <= zona[0] or  
                objeto[0] >= zona[2] or  
                objeto[3] <= zona[1] or  
                objeto[1] >= zona[3]) 
    else:
        # Verifica si el objeto está completamente dentro (versión original)
        return (objeto[0] >= zona[0] and
                objeto[1]+20 >= zona[1] and
                objeto[2] <= zona[2] and
                objeto[3] <= zona[3]+25)

box_annotator  = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(['#FF0000', '#0000FF', '#FFCC00']),
    thickness=2
)
mask_annotator = sv.MaskAnnotator(
    color=sv.ColorPalette.from_hex(['#009900', '#000000', '#FF0099']),
    opacity=0.2
)
ellipse_annotator = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(['#FF0000', '#0000FF', '#FFCC00']),
    thickness=2
)
triangle_annotator = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FFFF66'),
    base=25,
    height=21,
    outline_thickness=1
)
corner_annotator = sv.BoxCornerAnnotator(
     color=sv.Color.from_hex('#FFFF66'),
)
corner_annotatorTrue = sv.BoxCornerAnnotator(
     color=sv.Color.from_hex('#00913F'),
)



with video_sink:
    for frame in tqdm(frame_generator, total=video_info.total_frames): 
        
        result = player_model.infer(frame, device=settings.device, confidence=0.3)[0]
        detections = sv.Detections.from_inference(result)
        
        result = handler_model.infer(frame, confidence=0.3)[0]
        detections_handler = sv.Detections.from_inference(result)
        
        result = field_model.infer(frame, device=settings.device, confidence=0.3)[0]
        zones = sv.Detections.from_inference(result)
        
        
        

        ball_detections = detections[detections.class_id == settings.BALL_ID]
        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

        hoop_detections = detections[detections.class_id == settings.HOOP_ID]
        hoop_detections.xyxy = sv.pad_boxes(xyxy=hoop_detections.xyxy, px=10)

        all_detections = detections[detections.class_id != settings.BALL_ID]
        all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
        all_detections = tracker.update_with_detections(detections=all_detections)

        players_detections = all_detections[all_detections.class_id == settings.PLAYER_ID]
        referees_detections = all_detections[all_detections.class_id == settings.REFEREE_ID]

        players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
        players_detections.class_id = team_classifier.predict(players_crops)

        players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
        players_detections.class_id = team_classifier.predict(players_crops)
        
        
        
        try:
            ball_xyxy = ball_detections.xyxy[0]  
        except:
            pass
                
        try:
            hoop_xyxy = hoop_detections.xyxy[0]
        except:
            pass
        
        esta_dentro=False
        
        try:
            esta_dentro = is_inside(ball_xyxy, hoop_xyxy, partial_overlap=False)
        except:
            pass
        
        if esta_dentro:
            ball_in_hoop=True
        
        
        all_detections = sv.Detections.merge([
            players_detections, referees_detections])
        
        

        handler_detections = detections_handler[detections_handler.class_id == settings.HANDEL]
        
        if len(handler_detections) > 0:
            # Encontrar el índice de la detección con mayor confianza
            max_confidence_idx = int(handler_detections.confidence.argmax())
            
            # Seleccionar solo esa detección
            handler_detections = handler_detections[max_confidence_idx]
            
        try:
            handler_xyxy=handler_detections.xyxy[0]
        except:
            pass
        
        real_handler=[]

        try:
            esta_tirando= is_inside(handler_xyxy, ball_xyxy)
            if esta_tirando:
                real_handler= handler_detections        
        except:
            pass

            
        

        all_detections.class_id = all_detections.class_id.astype(int)

        annotated_frame = frame.copy()
        # annotated_frame = annotator.annotate_frame(
        #         annotated_frame, 
        #         all_detections, 
        #         ball_detections, 
        #         zones, 
        #         hoop_detections, esta_dentro
        #     )
        
        annotated_frame = ellipse_annotator.annotate(
            scene=annotated_frame,
            detections=all_detections)
        annotated_frame = triangle_annotator.annotate(
            scene=annotated_frame,
            detections=ball_detections)
        
        
        if esta_dentro:
            annotated_frame = corner_annotatorTrue.annotate(
                scene=annotated_frame,
                detections=hoop_detections)
        else:
            annotated_frame = corner_annotator.annotate(
                scene=annotated_frame,
                detections=hoop_detections)
        

        

        annotated_frame = mask_annotator.annotate(
            scene=annotated_frame,
            detections=zones)

        try:
            if zones.mask is not None and len(zones.mask) > 0:
                # Elige una clase específica de zona que quieras usar (ej: class_id == 1)
                target_class_id_1 = 1  # Ajusta este valor según tu necesidad
                target_class_id_2 = 2  # Ajusta este valor según tu necesidad
                
                # Filtra las zonas por class_id
                target_zones = zones[(zones.class_id == target_class_id_1) | (zones.class_id == target_class_id_2)]
                
                last_handler = True
                
                for mask in target_zones.mask:
                    polygons = sv.mask_to_polygons(mask)
                    
                    if len(polygons) > 0:
                        polygon_zone = sv.PolygonZone(polygon=polygons[0])
                        # Solo considerar detecciones de jugadores
                        players_only_detections = detections[detections.class_id == settings.PLAYER_ID]

                        if hasattr(real_handler, 'xyxy') and real_handler.xyxy.size > 0:
                            detections_in_zone = real_handler[polygon_zone.trigger(real_handler)]
                        
                        
                        if len(detections_in_zone.xyxy) > 0:
                            last_handler = False
                            


                if remaining_frames > 0:
                    remaining_frames -= 1
                    annotator.draw_nba_style_text(annotated_frame, active_text)
                    if remaining_frames == 0:
                        active_text = None
                        ball_in_hoop = False
                elif ball_in_hoop:
                    if last_handler:
                        active_text = "3Puntos"
                    else:
                        active_text = "2Puntos"
                    remaining_frames = settings.TEXT_DURATION_FRAMES
                    ball_in_hoop = False
                
        except Exception as e:
            print(e)
            pass
        # sv.plot_image(annotated_frame)
        video_sink.write_frame(annotated_frame)
