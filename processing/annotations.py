import supervision as sv
from config.constants import Colors, NBA
import cv2
import numpy as np
import time

class Annotator:
    def __init__(self):
        self.box_annotator = sv.BoxAnnotator(
            color=sv.ColorPalette.from_hex([Colors.TEAM_A, Colors.TEAM_B, Colors.REFEREE]),
            thickness=2
        )
        self.mask_annotator = sv.MaskAnnotator(
            color=sv.ColorPalette.from_hex([Colors.COURT, Colors.BOTTLE, Colors.AREA]),
            opacity=0.2
        )
        self.ellipse_annotator = sv.EllipseAnnotator(
            color=sv.ColorPalette.from_hex([Colors.TEAM_A, Colors.TEAM_B, Colors.REFEREE]),
            thickness=2
        )
        self.triangle_annotator = sv.TriangleAnnotator(
            color=sv.Color.from_hex(Colors.BALL),
            base=25,
            height=21,
            outline_thickness=1
        )
        self.corner_annotator = sv.BoxCornerAnnotator(
            color=sv.Color.from_hex(Colors.BALL),
        )
        self.corner_annotator_true = sv.BoxCornerAnnotator(
            color=sv.Color.from_hex('#00913F'),
        )
        
    def annotate_frames(self, annotated_frame, all_detections, ball_detections, zones, hoop_detections, esta_dentro):
        annotated_frame = self.ellipse_annotator.annotate(
            scene=annotated_frame,
            detections=all_detections)
        annotated_frame = self.triangle_annotator.annotate(
            scene=annotated_frame,
            detections=ball_detections)
        if esta_dentro:
            annotated_frame = self.corner_annotator_true.annotate(
                scene=annotated_frame,
                detections=hoop_detections)
        else:
            annotated_frame = self.corner_annotator.annotate(
                scene=annotated_frame,
                detections=hoop_detections)
        
        annotated_frame = self.mask_annotator.annotate(
            scene=annotated_frame,
            detections=zones)
        
        return annotated_frame
    
    def draw_nba_style_text(self, frame, text, center_x=None, center_y=None):
        """Dibuja texto estilo NBA con animaciones y efectos visuales."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 3
        
        if center_x is None:
            center_x = frame.shape[1] // 2
        if center_y is None:
            center_y = frame.shape[0] // 2
        
        elapsed_time = time.time() % 2
        if elapsed_time < 1.0:
            font_scale = 1 + 2 * elapsed_time
        else:
            font_scale = max(2.5, 3 - (elapsed_time - 1.0))
        
        # Sombra 3D
        shadow_color = (0, 0, 0)
        for offset in range(10, 0, -3):
            cv2.putText(
                frame, text,
                (center_x - offset, center_y + offset),
                font, font_scale,
                shadow_color,
                thickness + 2,
                cv2.LINE_AA,
            )
        
        # Texto principal
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = center_x - text_size[0] // 2
        text_y = center_y + text_size[1] // 2
        
        current_color = NBA.TEAM_COLORS[int(time.time() * 2) % 3]
        
        cv2.putText(
            frame, text,
            (text_x, text_y),
            font, font_scale,
            current_color,
            thickness + 3,
            cv2.LINE_AA,
        )
        
        # Línea decorativa
        cv2.line(
            frame,
            (text_x, text_y + 15),
            (text_x + text_size[0], text_y + 15),
            current_color,
            3,
            cv2.LINE_AA,
        )
        
        # Destellos aleatorios
        if elapsed_time > 1.0:
            for _ in range(20):
                x = np.random.randint(text_x, text_x + text_size[0])
                y = np.random.randint(text_y - text_size[1], text_y)
                cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)