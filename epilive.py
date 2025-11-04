# epilive_smooth.py
from ultralytics import YOLO
import cv2
import time
from datetime import datetime
import numpy as np

# Carrega o modelo treinado
model = YOLO("runs/detect/epi_treinamento/weights/best.pt")

# Dicionário para rastrear objetos entre frames
tracked_objects = {}
next_object_id = 0

def apply_nms(boxes, scores, iou_threshold=0.5):
    """Aplica Non-Maximum Suppression para remover detecções duplicadas"""
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    indices = np.argsort(scores)[::-1]
    
    keep = []
    
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
            
        xx1 = np.maximum(x1[current], x1[indices[1:]])
        yy1 = np.maximum(y1[current], y1[indices[1:]])
        xx2 = np.minimum(x2[current], x2[indices[1:]])
        yy2 = np.minimum(y2[current], y2[indices[1:]])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        intersection = w * h
        iou = intersection / (areas[current] + areas[indices[1:]] - intersection)
        
        remaining_indices = np.where(iou <= iou_threshold)[0]
        indices = indices[remaining_indices + 1]
    
    return keep

def smooth_boxes(current_detections, alpha=0.7):
    """Suaviza as bounding boxes entre frames"""
    global tracked_objects, next_object_id
    
    smoothed_detections = []
    
    for detection in current_detections:
        bbox = detection['bbox']
        class_name = detection['class_name']
        confidence = detection['confidence']
        
        # Centro e dimensões da bbox atual
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        # Procura objeto similar em frames anteriores
        best_match_id = None
        min_distance = float('inf')
        
        for obj_id, obj_data in tracked_objects.items():
            if obj_data['class_name'] != class_name:
                continue
                
            # Calcula distância entre centros
            prev_center_x = (obj_data['bbox'][0] + obj_data['bbox'][2]) / 2
            prev_center_y = (obj_data['bbox'][1] + obj_data['bbox'][3]) / 2
            distance = np.sqrt((center_x - prev_center_x)**2 + (center_y - prev_center_y)**2)
            
            # Se está próximo o suficiente, é o mesmo objeto
            if distance < 100:  # threshold de distância
                if distance < min_distance:
                    min_distance = distance
                    best_match_id = obj_id
        
        if best_match_id is not None:
            # 🔧 SUAVIZAÇÃO: Interpola com bbox anterior
            prev_bbox = tracked_objects[best_match_id]['bbox']
            
            # Suaviza coordenadas
            smooth_bbox = [
                alpha * prev_bbox[0] + (1 - alpha) * bbox[0],  # x1
                alpha * prev_bbox[1] + (1 - alpha) * bbox[1],  # y1
                alpha * prev_bbox[2] + (1 - alpha) * bbox[2],  # x2
                alpha * prev_bbox[3] + (1 - alpha) * bbox[3]   # y2
            ]
            
            # Atualiza objeto rastreado
            tracked_objects[best_match_id] = {
                'bbox': smooth_bbox,
                'class_name': class_name,
                'confidence': confidence,
                'last_seen': time.time()
            }
            
            smoothed_detections.append({
                'bbox': smooth_bbox,
                'class_name': class_name,
                'confidence': confidence,
                'object_id': best_match_id
            })
            
        else:
            # Novo objeto
            object_id = next_object_id
            next_object_id += 1
            
            tracked_objects[object_id] = {
                'bbox': bbox,
                'class_name': class_name,
                'confidence': confidence,
                'last_seen': time.time()
            }
            
            smoothed_detections.append({
                'bbox': bbox,
                'class_name': class_name,
                'confidence': confidence,
                'object_id': object_id
            })
    
    # Remove objetos antigos (não vistos há mais de 1 segundo)
    current_time = time.time()
    expired_objects = [
        obj_id for obj_id, obj_data in tracked_objects.items() 
        if current_time - obj_data['last_seen'] > 1.0
    ]
    for obj_id in expired_objects:
        del tracked_objects[obj_id]
    
    return smoothed_detections

def run_live_detection():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Erro: Não foi possível acessar a câmera")
        return
    
    print("Câmera inicializada - 20 segundos de teste")
    print("BBox Suavizada | NMS Ativado | Pressione 'q' para sair")
    print("Cores: VERDE=capacete | VERMELHO=sem_capacete")
    
    start_time = time.time()
    frame_count = 0
    
    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time >= 20:
            print(f"Tempo limite de 20 segundos atingido")
            break
        
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar frame")
            break
        
        frame = cv2.flip(frame, 1)
        frame_count += 1
        
        # Detecção
        results = model.predict(
            source=frame,
            conf=0.45,
            iou=0.5,  # NMS do YOLO
            verbose=False
        )
        
        result = results[0]
        display_frame = frame.copy()
        
        current_detections = []
        class_counts = {'capacete': 0, 'sem_capacete': 0}
        
        if result.boxes:
            boxes = []
            scores = []
            class_ids = []
            confidences = []
            
            for box in result.boxes:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                bbox = box.xyxy[0].tolist()
                
                if confidence > 0.45:
                    boxes.append(bbox)
                    scores.append(confidence)
                    class_ids.append(class_id)
                    confidences.append(confidence)
            
            # Aplica NMS
            if len(boxes) > 0:
                keep_indices = apply_nms(boxes, scores, iou_threshold=0.5)
                
                for idx in keep_indices:
                    class_id = class_ids[idx]
                    class_name = model.names[class_id]
                    confidence = confidences[idx]
                    bbox = boxes[idx]
                    
                    current_detections.append({
                        'bbox': bbox,
                        'class_name': class_name,
                        'confidence': confidence
                    })
        
        # 🔧 SUAVIZAÇÃO: Aplica suavização nas bounding boxes
        if current_detections:
            smoothed_detections = smooth_boxes(current_detections, alpha=0.7)
        else:
            smoothed_detections = []
        
        # Desenha bounding boxes suavizadas
        for detection in smoothed_detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            class_counts[class_name] += 1
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Define cores
            if class_name == 'capacete':
                color = (0, 255, 0)  # VERDE
                label = f"Capacete: {confidence:.1%}"
            else:  # sem_capacete
                color = (0, 0, 255)  # VERMELHO
                label = f"SEM CAPACETE: {confidence:.1%}"
            
            # Desenha bbox suavizada
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(display_frame, label, 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, color, 2)
        
        # Informações na tela
        cv2.putText(display_frame, f"Tempo: {20 - int(elapsed_time):2d}s", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Frame: {frame_count}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Capacetes: {class_counts['capacete']}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Sem Capacetes: {class_counts['sem_capacete']}", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(display_frame, "BBox SUAVIZADA - Sem pulsar", 
                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        cv2.imshow('EPI Detection - BBox Suavizada', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("⏹️  Interrompido pelo usuário")
            break
    
    # Estatísticas finais
    print(f"\nRELATÓRIO FINAL:")
    print(f"Tempo total: {elapsed_time:.1f} segundos")
    print(f"Capacetes: {class_counts['capacete']}")
    print(f"Sem capacete: {class_counts['sem_capacete']}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_detection()