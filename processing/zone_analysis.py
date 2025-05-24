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

def analyze_shooting_zone(zones, real_handler):
    """
    Analiza si el tiro es de 2 o 3 puntos basado en la posición del jugador.
    
    Returns:
        str or None: "2Puntos", "3Puntos" o None si no hay tiro
    """
    try:
        if zones.mask is None or len(zones.mask) == 0:
            return None
            
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

                if hasattr(real_handler, 'xyxy') and real_handler.xyxy.size > 0:
                    detections_in_zone = real_handler[polygon_zone.trigger(real_handler)]
                                
                                
                if len(detections_in_zone.xyxy) > 0:
                    last_handler = False
        
        # if is_three_point and hasattr(ball_detections, 'xyxy') and ball_detections.xyxy.size > 0:
        #     return "3Puntos"
        # elif not is_three_point and hasattr(ball_detections, 'xyxy') and ball_detections.xyxy.size > 0:
        #     return "2Puntos"
        if last_handler :
            return "3Puntos"
        else:
            return "2Puntos"
    except:
        pass
    
def real_handler_detections(ball_detections, hoop_detections, handler_detections):
    real_handler=[]
    esta_dentro=False
    
    try:
        ball_xyxy = ball_detections.xyxy[0]  
        print(ball_xyxy)
    except:
        pass
                
    try:
        hoop_xyxy = hoop_detections.xyxy[0]
    except:
        pass
        
        
    try:
        esta_dentro = is_inside(ball_xyxy, hoop_xyxy, partial_overlap=False)
    except:
        pass
        
        
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
        if esta_tirando:
                real_handler= handler_detections        
    except:
        pass
    
    return esta_dentro, real_handler