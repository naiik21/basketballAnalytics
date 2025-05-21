import numpy as np
import supervision as sv

def is_inside(objeto, zona, partial_overlap=True):
    """
    Verifica si el objeto está dentro o solapando la zona.
    
    Args:
        objeto (np.ndarray): Bounding box del objeto [x1, y1, x2, y2]
        zona (np.ndarray): Bounding box de la zona [x1, y1, x2, y2]
        partial_overlap (bool): Si True, devuelve True si hay superposición parcial.
    
    Returns:
        bool: True si el objeto está dentro/solapando, False en caso contrario.
    """
    if len(objeto.shape) > 1:
        objeto = objeto[0]
    if len(zona.shape) > 1:
        zona = zona[0]
            
    if partial_overlap:
        return not (objeto[2] <= zona[0] or  
                   objeto[0] >= zona[2] or  
                   objeto[3] <= zona[1] or  
                   objeto[1] >= zona[3]) 
    else:
        return (objeto[0] >= zona[0] and
                objeto[1]+20 >= zona[1] and
                objeto[2] <= zona[2] and
                objeto[3] <= zona[3]+25)

def analyze_shooting_zone(frame, zones, handler_detections, ball_detections):
    """
    Analiza si el tiro es de 2 o 3 puntos basado en la posición del jugador.
    
    Returns:
        str or None: "2Puntos", "3Puntos" o None si no hay tiro
    """
    if zones.mask is None or len(zones.mask) == 0:
        return None
        
    target_class_id_1 = 1  # Zona de 2 puntos
    target_class_id_2 = 2  # Zona de 3 puntos
    
    target_zones = zones[(zones.class_id == target_class_id_1) | (zones.class_id == target_class_id_2)]
    
    is_three_point = True
    
    for mask in target_zones.mask:
        polygons = sv.mask_to_polygons(mask)
        
        if len(polygons) > 0:
            polygon_zone = sv.PolygonZone(polygon=polygons[0])
            
            if hasattr(handler_detections, 'xyxy') and handler_detections.xyxy.size > 0:
                detections_in_zone = handler_detections[polygon_zone.trigger(handler_detections)]
                
                if len(detections_in_zone.xyxy) > 0:
                    is_three_point = False
    
    if is_three_point and hasattr(ball_detections, 'xyxy') and ball_detections.xyxy.size > 0:
        return "3Puntos"
    elif not is_three_point and hasattr(ball_detections, 'xyxy') and ball_detections.xyxy.size > 0:
        return "2Puntos"
    
    return None