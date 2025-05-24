import numpy as np

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