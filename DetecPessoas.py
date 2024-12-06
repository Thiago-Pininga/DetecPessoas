import tensorflow as tf
import cv2
import numpy as np
import kagglehub

# Baixar o modelo utilizando o kagglehub
MODEL_PATH = kagglehub.model_download("tensorflow/ssd-mobilenet-v2/tensorFlow2/ssd-mobilenet-v2")
print("Modelo baixado para o caminho:", MODEL_PATH)

# Carregar o modelo pré-treinado do TensorFlow (MobileNet-SSD)
detect_fn = tf.saved_model.load(MODEL_PATH)

# Definir a classe para detectar "person"
category_index = {
    1: "person"
}

# Função para processar o frame e detectar pessoas
def detect_people(frame):
    height, width, _ = frame.shape
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]  # Adiciona dimensão de lote

    detections = detect_fn(input_tensor)

    boxes = detections["detection_boxes"][0].numpy()
    scores = detections["detection_scores"][0].numpy()
    classes = detections["detection_classes"][0].numpy().astype(int)

    count = 0

    for i in range(len(scores)):
        if scores[i] > 0.5 and category_index.get(classes[i]) == "person":  # Verifica se a detecção é de pessoa
            count += 1
            box = boxes[i]
            y1, x1, y2, x2 = int(box[0] * height), int(box[1] * width), int(box[2] * height), int(box[3] * width)

            # Desenha a caixa delimitadora no frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Pessoa", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame, count

# Captura vídeo da câmera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao acessar a câmera.")
    exit()

print("Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar o frame.")
        break

    # Detectar pessoas no frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    processed_frame, count = detect_people(frame_rgb)

    # Exibir contagem de pessoas detectadas
    cv2.putText(processed_frame, f"Pessoas detectadas: {count}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Alerta de sobrecarga de pessoas (se mais de 2 pessoas detectadas)
    if count >= 3:  # Sobrecarga se houver 3 ou mais pessoas
        cv2.putText(processed_frame, "Sobrecarga de pessoas!", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Exibe o frame com a contagem e o alerta (se houver)
    cv2.imshow("Detecção de Pessoas", processed_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Interrompe quando pressionado 'q'
        break

cap.release()  # Libera a captura da câmera
cv2.destroyAllWindows()  # Fecha todas as janelas do OpenCV
