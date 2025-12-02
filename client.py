#!/usr/bin/env python3
# client.py — Device (HP) : capture caméra, pré-détecte/ROI, compression adaptative, envoi avec metadata, reconnexion

import cv2
import socket
import json
import struct
import time
import numpy as np

SERVER_IP = "192.168.1.3"   # <-- IP du PC Dell
SERVER_PORT = 5000

# Paramètres reconnection
MAX_RETRIES = 10
INITIAL_BACKOFF = 1.0  # sec

# Paramètres detection préliminaire (simple motion/contour)
MIN_CONTOUR_AREA = 2000  # ajuster selon la caméra / distance
ROI_PADDING = 10         # pixels de padding autour du bbox

# Paramètres JPEG adaptatif
QUALITY_HIGH = 85
QUALITY_LOW = 45
MOTION_THRESHOLD = 1000  # somme des pixels détectés pour considérer 'beaucoup de mouvement'

# Helper : envoie tous les octets
def send_all(sock, data: bytes):
    total_sent = 0
    while total_sent < len(data):
        sent = sock.send(data[total_sent:])
        if sent == 0:
            raise RuntimeError("socket connection broken")
        total_sent += sent

# Encode metadata (json) and image bytes with protocol:
# [4 bytes meta_len][meta_json][8 bytes img_size][img_bytes]
def send_image_with_metadata(sock, img, metadata: dict, jpeg_quality: int):
    ok, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise RuntimeError("Erreur encodage image")
    img_bytes = buffer.tobytes()

    meta_json = json.dumps(metadata).encode('utf-8')
    meta_len = struct.pack('>I', len(meta_json))           # 4 bytes big-endian
    img_size = struct.pack('>Q', len(img_bytes))           # 8 bytes big-endian

    send_all(sock, meta_len)
    send_all(sock, meta_json)
    send_all(sock, img_size)
    send_all(sock, img_bytes)

def connect_with_backoff():
    backoff = INITIAL_BACKOFF
    retries = 0
    while retries < MAX_RETRIES:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((SERVER_IP, SERVER_PORT))
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"[CLIENT] Connecté au serveur {SERVER_IP}:{SERVER_PORT}")
            return s
        except Exception as e:
            print(f"[CLIENT] Erreur connexion : {e} — retry dans {backoff:.1f}s")
            time.sleep(backoff)
            backoff *= 2
            retries += 1
    raise RuntimeError("Impossible de se connecter après plusieurs tentatives")

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Erreur : caméra introuvable")
        return

    sock = None
    try:
        sock = connect_with_backoff()
    except Exception as e:
        print("Abandon connexion :", e)
        
        return

    prev_gray = None
    frame_count = 0

    try:
        while True:
            frame_count += 1
            ret, frame = cap.read()
            if not ret:
                print("Erreur capture image")
                break

            # Resize raisonnable (pour réduire taille)
            max_w = 800
            h, w = frame.shape[:2]
            if w > max_w:
                scale = max_w / w
                frame = cv2.resize(frame, (int(w*scale), int(h*scale)))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.GaussianBlur(gray, (5,5), 0)

            # Motion detection simple : différence avec frame précédente
            motion_pixels = 0
            roi = None
            roi_box = None
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray_blur)
                _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                motion_pixels = int(np.sum(th) / 255)  # nb pixels en mouvement

                # Find contours and decide ROI if large contour exist
                contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                max_area = 0
                max_cnt = None
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > max_area:
                        max_area = area
                        max_cnt = cnt

                if max_area > MIN_CONTOUR_AREA and max_cnt is not None:
                    x, y, ww, hh = cv2.boundingRect(max_cnt)
                    # add padding and clamp
                    x0 = max(0, x-ROI_PADDING)
                    y0 = max(0, y-ROI_PADDING)
                    x1 = min(frame.shape[1], x+ww+ROI_PADDING)
                    y1 = min(frame.shape[0], y+hh+ROI_PADDING)
                    roi_box = (x0, y0, x1, y1)
                    roi = frame[y0:y1, x0:x1]

            # Choix qualité JPEG adaptative
            if motion_pixels > MOTION_THRESHOLD:
                quality = QUALITY_HIGH
            else:
                quality = QUALITY_LOW

            # Préparer metadata
            metadata = {
                "device_id": "HP_node_01",
                "timestamp": time.time(),
                "frame_index": frame_count,
                "motion_pixels": motion_pixels,
            }

            # Si on a un ROI valable, on envoie seulement le crop + bbox info
            if roi is not None:
                metadata["roi_sent"] = True
                metadata["bbox"] = {"x0": roi_box[0], "y0": roi_box[1], "x1": roi_box[2], "y1": roi_box[3]}
                img_to_send = roi
            else:
                metadata["roi_sent"] = False
                metadata["bbox"] = None
                img_to_send = frame

            # Tentative d'envoi, avec reconnexion si nécessaire
            try:
                send_image_with_metadata(sock, img_to_send, metadata, quality)
            except Exception as e:
                print("[CLIENT] Erreur envoi:", e)
                # tenter reconnexion
                try:
                    sock.close()
                except:
                    pass
                print("[CLIENT] Tentative de reconnexion...")
                try:
                    sock = connect_with_backoff()
                    # réessayer une fois
                    send_image_with_metadata(sock, img_to_send, metadata, quality)
                except Exception as e2:
                    print("[CLIENT] Reconnexion ou ré-envoi échoué :", e2)
                    break

            # Lire réponse (simple : serveur renvoie texte)
            try:
                resp = sock.recv(64)
                if not resp:
                    raise RuntimeError("socket closed by server")
                print(f"[CLIENT] Réponse serveur : {resp.decode().strip()}")
            except Exception as e:
                print("[CLIENT] Erreur lecture réponse:", e)
                # essayer reconnexion la prochaine boucle
                try:
                    sock.close()
                except:
                    pass
                try:
                    sock = connect_with_backoff()
                except Exception as e2:
                    print("[CLIENT] Reconnexion échouée :", e2)
                    break

            prev_gray = gray_blur

            # Pause entre captures — ajustable
            time.sleep(0.5)

    finally:
        cap.release()
        if sock:
            sock.close()
        print("[CLIENT] Terminé")

if __name__ == "__main__":
    main()
