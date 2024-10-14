import cv2
import mediapipe as mp

# Inicialização dos módulos do MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_hand_tracking = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Inicializa o detector de rosto e de mãos
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)
hands = mp_hand_tracking.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Captura de vídeo da webcam
cap = cv2.VideoCapture(0)

# Função para desenhar os pontos e as conexões pontilhadas
def draw_hand_landmarks(image, hand_landmarks):
    h, w, _ = image.shape
    # Desenha os pontos dos dedos
    for landmark in hand_landmarks.landmark:
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(image, (x, y), 5, (255, 0, 0), -1)
    
    # Conecta os pontos com linhas pontilhadas
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Polegar
        (5, 6), (6, 7), (7, 8),           # Indicador
        (9, 10), (10, 11), (11, 12),      # Médio
        (13, 14), (14, 15), (15, 16),     # Anular
        (17, 18), (18, 19), (19, 20)      # Mínimo
    ]
    
    for connection in connections:
        start_idx, end_idx = connection
        start = hand_landmarks.landmark[start_idx]
        end = hand_landmarks.landmark[end_idx]
        start_point = (int(start.x * w), int(start.y * h))
        end_point = (int(end.x * w), int(end.y * h))
        cv2.line(image, start_point, end_point, (255, 0, 0), 2, cv2.LINE_AA)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Converte o frame para RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detecta mãos
    hand_results = hands.process(rgb_frame)
    
    # Desenha as mãos e os dedos detectados
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            draw_hand_landmarks(frame, hand_landmarks)
    
    # Exibe o frame com as deteções
    cv2.imshow('Face and Hand Detection', frame)
    
    # Sai do loop ao pressionar a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()