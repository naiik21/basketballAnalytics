import cv2
import os
import argparse

def extract_frames(video_path, output_folder, frame_interval=1):
    """
    Extrae frames de un video y los guarda en una carpeta.
    
    Args:
        video_path (str): Ruta al archivo de video.
        output_folder (str): Carpeta donde se guardarán los frames.
        frame_interval (int): Cada cuántos frames se extrae una imagen (1 = todos los frames).
    """
    # Crear la carpeta de salida si no existe
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Abrir el video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: No se pudo abrir el video.")
        return
    
    # Obtener información del video
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Video info:")
    print(f"- Ruta: {video_path}")
    print(f"- Resolución: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"- FPS: {fps:.2f}")
    print(f"- Total frames: {total_frames}")
    print(f"- Duración: {duration:.2f} segundos")
    print(f"- Extrayendo 1 frame cada {frame_interval} frames...")
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        
        # Si no hay más frames, salir del bucle
        if not ret:
            break
        
        # Guardar frame si coincide con el intervalo
        if frame_count % frame_interval == 0:
            frame_name = os.path.join(output_folder, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_name, frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"\nExtracción completada!")
    print(f"- Frames procesados: {frame_count}")
    print(f"- Frames guardados: {saved_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extrae frames de un video.')
    parser.add_argument('video', help='Ruta al archivo de video')
    parser.add_argument('--output', default='frames', help='Carpeta de salida (por defecto: frames)')
    parser.add_argument('--interval', type=int, default=1, help='Intervalo de frames a extraer (por defecto: 1)')
    
    args = parser.parse_args()
    
    extract_frames(args.video, args.output, args.interval)
    
    
# python extract_frames.py mi_video.mp4 --output mis_frames --interval 5