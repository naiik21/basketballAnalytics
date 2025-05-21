import os
import torch
import time 
import supervision as sv
import numpy as np
import cv2
from dotenv import load_dotenv
from tqdm import tqdm
from inference import get_model
from models.teamClassifier import TeamClassifier

load_dotenv()

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

PLAYER_DETECTION_MODEL_ID="basketball-players-fy4c2/25"
PLAYER_MODEL_DETECTION=get_model(model_id=PLAYER_DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY)

FIELD_DETECTION_MODEL_ID="basketball_court_segmentation-tlfja/5"
FIELD_MODEL_DETECTION=get_model(model_id=FIELD_DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY)

HANDLER_DETECTION_MODEL_ID="basketball_and_hoops/3"
HANDLER_MODEL_DETECTION=get_model(model_id=HANDLER_DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

SOURCE_VIDEO_PATH="./videos/test/stepback.mp4"
TAGET_VIDEO_PATH="./videos/result/shastepback_result_2.mp4"

STRIDE=30

HANDEL=0
BALL_ID = 1
HOOP_ID = 2
PLAYER_ID = 4
REFEREE_ID = 5

frame_generator = sv.get_video_frames_generator(
    source_path=SOURCE_VIDEO_PATH, stride=STRIDE)

crops = []
for frame in tqdm(frame_generator, desc='collecting crops'):
    result = PLAYER_MODEL_DETECTION.infer(frame, device=device, confidence=0.3)[0]
    detections = sv.Detections.from_inference(result)
    detections = detections.with_nms(threshold=0.5, class_agnostic=True)
    detections = detections[detections.class_id == PLAYER_ID]
    players_crops = [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]
    crops += players_crops

team_classifier = TeamClassifier(device="cuda")


team_classifier.fit(crops)

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

def draw_nba_style_text(frame, text, center_x=None, center_y=None, duration=30):
    """
    Dibuja texto estilo NBA (animación, sombra 3D, colores vibrantes).
    
    Args:
        frame (np.ndarray): Imagen donde dibujar el texto.
        text (str): Texto a mostrar (ej: "3Puntos").
        center_x (int): Posición X central (si None, se centra automáticamente).
        center_y (int): Posición Y central (si None, se centra automáticamente).
        duration (int): Duración en frames de la animación (opcional).
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 3
    
    # Si no se especifica centro, usa el centro del frame
    if center_x is None:
        center_x = frame.shape[1] // 2
    if center_y is None:
        center_y = frame.shape[0] // 2
    
    # ---- Efecto de animación (crece y se estabiliza) ----
    elapsed_time = time.time() % 2  # Ciclo de 2 segundos para la animación
    if elapsed_time < 1.0:
        font_scale = 1 + 2 * elapsed_time  # Escala de 1 a 3
    else:
        font_scale = max(2.5, 3 - (elapsed_time - 1.0))  # Rebote suave
    
    # ---- Sombra 3D (3 capas desplazadas) ----
    shadow_color = (0, 0, 0)  # Negro
    for offset in range(10, 0, -3):
        cv2.putText(
            frame, text,
            (center_x - offset, center_y + offset),
            font, font_scale,
            shadow_color,
            thickness + 2,
            cv2.LINE_AA,
        )
    
    # ---- Texto principal con gradiente de color ----
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = center_x - text_size[0] // 2
    text_y = center_y + text_size[1] // 2
    
    # Ciclo de colores NBA (rojo, azul, dorado)
    current_color = [
        (0, 0, 255),    # Rojo
        (255, 200, 0),  # Dorado
        (0, 150, 255),  # Azul claro
    ][int(time.time() * 2) % 3]  # Cambia cada 0.5 segundos
    
    cv2.putText(
        frame, text,
        (text_x, text_y),
        font, font_scale,
        current_color,
        thickness + 3,
        cv2.LINE_AA,
    )
    
    # ---- Línea decorativa inferior ----
    cv2.line(
        frame,
        (text_x, text_y + 15),
        (text_x + text_size[0], text_y + 15),
        current_color,
        3,
        cv2.LINE_AA,
    )
    
    # ---- Destellos aleatorios (solo si la animación está estable) ----
    if elapsed_time > 1.0:
        for _ in range(20):
            x = np.random.randint(text_x, text_x + text_size[0])
            y = np.random.randint(text_y - text_size[1], text_y)
            cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

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
label_annotator = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(['#FF0000', '#0000FF', '#FFCC00']),
    text_color=sv.Color.from_hex('#000000'),
    text_position=sv.Position.BOTTOM_CENTER
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


video_info= sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
video_sink=sv.VideoSink(TAGET_VIDEO_PATH, video_info=video_info)
frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

TEXT_DURATION_FRAMES = 30  # Número de frames que el texto permanecerá visible
active_text = None  # Guarda el texto actual ("3Puntos" o "2Puntos")
remaining_frames = 0  # Frames restantes para ocultar el texto

tracker = sv.ByteTrack()
tracker.reset()

aux= False
last_handler=True
with video_sink:
    for frame in tqdm(frame_generator, total=video_info.total_frames): 
        result = PLAYER_MODEL_DETECTION.infer(frame, device=device, confidence=0.3)[0]
        detections = sv.Detections.from_inference(result)

        ball_detections = detections[detections.class_id == BALL_ID]
        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

        hoop_detections = detections[detections.class_id == HOOP_ID]
        hoop_detections.xyxy = sv.pad_boxes(xyxy=hoop_detections.xyxy, px=10)

        all_detections = detections[detections.class_id != BALL_ID]
        all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
        all_detections = tracker.update_with_detections(detections=all_detections)

        players_detections = all_detections[all_detections.class_id == PLAYER_ID]
        referees_detections = all_detections[all_detections.class_id == REFEREE_ID]

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
            aux=True
        
        
        all_detections = sv.Detections.merge([
            players_detections, referees_detections])
        
        result = HANDLER_MODEL_DETECTION.infer(frame, confidence=0.3)[0]
        detections_handler = sv.Detections.from_inference(result)

        handler_detections = detections_handler[detections_handler.class_id == HANDEL]
        
        if len(handler_detections) > 0:
            # Encontrar el índice de la detección con mayor confianza
            max_confidence_idx = int(handler_detections.confidence.argmax())
            
            # Seleccionar solo esa detección
            handler_detections = handler_detections[max_confidence_idx]
            
        try:
            handler_xyxy=handler_detections.xyxy[0]
        except:
            pass

        try:
            esta_tirando= is_inside(handler_xyxy, ball_xyxy)
        except:
            pass
        real_handler=[]
    
        
        if esta_tirando:
            real_handler= handler_detections
            
            
        labels = [
            f"#{tracker_id}"
            for tracker_id
            in all_detections.tracker_id
        ]

        all_detections.class_id = all_detections.class_id.astype(int)

        annotated_frame = frame.copy()
        annotated_frame = ellipse_annotator.annotate(
            scene=annotated_frame,
            detections=all_detections)
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=all_detections,
            labels=labels)
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
        

        result = FIELD_MODEL_DETECTION.infer(frame, device=device, confidence=0.3)[0]
        zones = sv.Detections.from_inference(result)

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
                        players_only_detections = detections[detections.class_id == PLAYER_ID]

                        if hasattr(real_handler, 'xyxy') and real_handler.xyxy.size > 0:
                            detections_in_zone = real_handler[polygon_zone.trigger(real_handler)]
                        
                        
                        if len(detections_in_zone.xyxy) > 0:
                            last_handler = False
                            


                if remaining_frames > 0:
                    remaining_frames -= 1
                    draw_nba_style_text(annotated_frame, active_text)  # <-- Nueva función
                    if remaining_frames == 0:
                        active_text = None
                        aux = False
                elif aux:
                    if last_handler:
                        active_text = "3Puntos"
                    else:
                        active_text = "2Puntos"
                    remaining_frames = TEXT_DURATION_FRAMES
                    aux = False
                
        except Exception as e:
            print(e)
            pass
        # sv.plot_image(annotated_frame)
        video_sink.write_frame(annotated_frame)
