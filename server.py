#!/usr/bin/env python3
import socket
import struct
import json
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
import traceback
import mediapipe as mp
import time

# Charger modèle YOLOv8-pose
model = YOLO('yolov8x-pose.pt')

SERVER_IP = '0.0.0.0'
SERVER_PORT = 5000
MAX_WORKERS = 8
CLIENT_SOCKET_TIMEOUT = 60  # secondes

# MediaPipe Hands init (réutilisable)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_hands_proc = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

# Seuil confiance keypoints
KP_CONF_THRESHOLD = 0.2

def recv_n(sock, n):
    data = b''
    while len(data) < n:
        try:
            packet = sock.recv(n - len(data))
        except socket.timeout:
            return None
        if not packet:
            return None
        data += packet
    return data

def safe_extract_keypoints(r, img_shape=None):
    """
    Tente d'extraire un numpy array shape (N,3) des keypoints de r.keypoints.
    Retourne None si pas de keypoints exploitables.
    img_shape: (h,w) si on doit convertir de coords normalisées en pixels.
    """
    kp_obj = getattr(r, 'keypoints', None)
    if kp_obj is None:
        return None

    # 1) Si l'objet expose .data (ultralytics Keypoints)
    data = getattr(kp_obj, 'data', None)
    conf = getattr(kp_obj, 'conf', None)
    orig_shape = getattr(kp_obj, 'orig_shape', None)

    try:
        # si data existe et n'est pas vide : tensor ou numpy
        if data is not None:
            try:
                # tente conversion en numpy (gère torch.Tensor aussi)
                arr = np.asarray(data)
            except Exception:
                # si c'est un tensor torch
                try:
                    import torch
                    if isinstance(data, torch.Tensor):
                        arr = data.cpu().numpy()
                    else:
                        arr = np.array(data)
                except Exception:
                    arr = None
            if arr is None:
                return None
            # arr shape possible: (num_detections, K, 3) or (K,3)
            if arr.ndim == 3 and arr.shape[0] >= 1:
                # prendre la première détection
                kp = arr[0]
            elif arr.ndim == 2 and arr.shape[1] >= 2:
                kp = arr
            else:
                return None

            # appliquer seuil de confiance si conf disponible
            if conf is not None:
                try:
                    conf_arr = np.asarray(conf)
                    if conf_arr.ndim == 2 and conf_arr.shape[0] >= 1:
                        conf_vec = conf_arr[0] if conf_arr.shape[0] > 1 else conf_arr
                        # masque keypoints faibles
                        mask = conf_vec >= KP_CONF_THRESHOLD
                        # si trop peu de keypoints valides, on considère invalide
                        if np.sum(mask) < 5:
                            return None
                        # remplacer keypoints faibles par NaN pour éviter erreurs
                        kp = np.asarray(kp, dtype=np.float32)
                        for i in range(kp.shape[0]):
                            if i >= mask.shape[0] or not mask[i]:
                                kp[i] = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
                except Exception:
                    pass

            # Certains modèles donnent coords normalisées (0..1) ; détecter et convertir en pixels si possible
            target_shape = None
            if orig_shape:
                try:
                    target_shape = (int(orig_shape[0]), int(orig_shape[1]))
                except Exception:
                    target_shape = None
            if target_shape is None and img_shape is not None:
                target_shape = img_shape

            if target_shape is not None:
                h, w = target_shape
                # si coords x,y dedans [0,1], convertir
                x_vals = kp[:, 0]
                # utiliser nanmax pour tolérer NaN
                try:
                    max_x = np.nanmax(np.abs(x_vals))
                except Exception:
                    max_x = 10.0
                if max_x <= 1.0:
                    # convertir normalisé->pixels
                    kp_pixels = kp.copy().astype(np.float32)
                    kp_pixels[:, 0] = kp[:, 0] * w
                    kp_pixels[:, 1] = kp[:, 1] * h
                    # z scale approximatif: multiplier par w
                    kp_pixels[:, 2] = kp[:, 2] * w
                    return kp_pixels
            return np.asarray(kp, dtype=np.float32)

        # 2) Sinon tenter conversion directe de r.keypoints (liste, tensor, etc.)
        try:
            arr = np.asarray(kp_obj, dtype=np.float32)
            if arr.size == 0:
                return None
            if arr.ndim == 3 and arr.shape[0] >= 1:
                return arr[0]
            if arr.ndim == 2:
                return arr
        except Exception:
            pass
    except Exception:
        return None

    return None

def count_fingers_from_landmarks(landmarks):
    """
    landmarks: numpy array (N,3) coords en pixels [x,y,z] ou normalisés (mais de préférence pixels).
    Méthode:
     - pour index/middle/ring/pinky: tip_y < pip_y -> doigt levé (coord y top-left).
     - pour le pouce: on compare tip_x vs ref_x selon orientation main (droite/gauche).
    Retourne int 0..5
    """
    if landmarks is None:
        return 0
    try:
        lm = np.asarray(landmarks, dtype=np.float32)
    except Exception:
        return 0
    if lm.ndim != 2 or lm.shape[0] < 5:
        return 0

    # Indices usuels (MediaPipe / conventions communes)
    TIP = { 'thumb':4, 'index':8, 'middle':12, 'ring':16, 'pinky':20 }
    PIP = { 'index':6, 'middle':10, 'ring':14, 'pinky':18 }  # PIP joints

    fingers = 0

    # vérifier orientation main: utiliser MCP/WRIST pour approx
    try:
        wrist_x = lm[0,0]
        index_mcp_x = lm[5,0] if lm.shape[0] > 5 else None
        hand_dir = 'right' if (index_mcp_x is not None and index_mcp_x > wrist_x) else 'left'
    except Exception:
        hand_dir = 'right'

    # doigts (index..pinky) : tip_y < pip_y
    for name in ['index','middle','ring','pinky']:
        tip_idx = TIP[name]
        pip_idx = PIP[name]
        if tip_idx >= lm.shape[0] or pip_idx >= lm.shape[0]:
            continue
        tip = lm[tip_idx]
        pip = lm[pip_idx]
        if np.isnan(tip).any() or np.isnan(pip).any():
            continue
        # top-left origin -> plus petit y = plus haut
        if tip[1] < pip[1] - 5:  # marge 5 pixels pour stabilité
            fingers += 1

    # pouce (utiliser x)
    t_idx = TIP['thumb']
    if t_idx < lm.shape[0]:
        thumb = lm[t_idx]
        if 2 < lm.shape[0]:
            ref = lm[2]
        else:
            ref = None
        if ref is not None and not np.isnan(thumb).any() and not np.isnan(ref).any():
            if hand_dir == 'right':
                if thumb[0] > ref[0] + 5:
                    fingers += 1
            else:
                if thumb[0] < ref[0] - 5:
                    fingers += 1

    return fingers

def count_fingers_mediapipe_from_image(img):
    """
    Utilise MediaPipe Hands pour détecter mains + landmarks et compter doigts.
    img: BGR numpy array (cv2)
    Retour: nombre maximal de doigts détectés parmi mains trouvées
    """
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = mp_hands_proc.process(img_rgb)
        if not results.multi_hand_landmarks:
            return 0
        max_fingers = 0
        h, w = img.shape[:2]
        for hand_landmarks in results.multi_hand_landmarks:
            lm = []
            for p in hand_landmarks.landmark:
                lm.append([p.x * w, p.y * h, p.z * w])
            lm = np.array(lm, dtype=np.float32)
            fingers = count_fingers_from_landmarks(lm)
            if fingers > max_fingers:
                max_fingers = fingers
        return max_fingers
    except Exception as e:
        print("[SERVER][MP] Erreur MediaPipe:", e)
        return 0

def handle_client(client_socket, addr):
    print(f"[SERVER] Client connecté : {addr}")
    client_socket.settimeout(CLIENT_SOCKET_TIMEOUT)
    try:
        while True:
            # Lire 4 bytes : longueur metadata
            meta_len_bytes = recv_n(client_socket, 4)
            if not meta_len_bytes:
                print(f"[SERVER] Client {addr} a fermé la connexion (meta_len)")
                break
            meta_len = struct.unpack('>I', meta_len_bytes)[0]

            meta_json = recv_n(client_socket, meta_len)
            if not meta_json:
                print(f"[SERVER] Client {addr} a fermé la connexion (meta_json)")
                break
            try:
                metadata = json.loads(meta_json.decode('utf-8'))
            except Exception:
                metadata = {}

            # Lire 8 bytes : taille image
            img_size_bytes = recv_n(client_socket, 8)
            if not img_size_bytes:
                print(f"[SERVER] Client {addr} a fermé la connexion (img_size)")
                break
            img_size = struct.unpack('>Q', img_size_bytes)[0]

            # Lire image
            img_bytes = recv_n(client_socket, img_size)
            if img_bytes is None:
                print(f"[SERVER] Client {addr} a fermé la connexion (img_bytes)")
                break

            # Décoder image
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                print(f"[SERVER] Erreur decodage image depuis {addr}")
                try:
                    client_socket.send(b"0")
                except:
                    pass
                continue

            device_id = metadata.get("device_id", "unknown")
            frame_index = metadata.get("frame_index", -1)
            motion_pixels = metadata.get("motion_pixels", 0)

            fingers = 0
            try:
                results = model(img)  # inference YOLO
                found_valid_hand = False

                for r in results:
                    # debug minimal (repr truncated)
                    try:
                        kp_repr = repr(getattr(r, 'keypoints', None))
                        # print a short slice to avoid flood
                        print(f"[SERVER][DEBUG] r.keypoints repr: {kp_repr[:200] if kp_repr else 'None'}")
                    except Exception:
                        pass

                    kp_arr = safe_extract_keypoints(r, img_shape=(img.shape[0], img.shape[1]))
                    if kp_arr is None:
                        continue

                    # show shape if present
                    print(f"[SERVER][DEBUG] extracted kp_arr shape: {getattr(kp_arr,'shape', None)}")

                    if kp_arr.ndim >= 2 and kp_arr.shape[0] >= 21:
                        fingers = count_fingers_from_landmarks(kp_arr)
                        found_valid_hand = True
                        break

                if not found_valid_hand:
                    # fallback: mediapipe on roi or entire image
                    # if client sent bbox, prefer crop
                    if metadata.get("roi_sent", False) and metadata.get("bbox"):
                        bbox = metadata["bbox"]
                        try:
                            x0 = int(bbox["x0"]); y0 = int(bbox["y0"])
                            x1 = int(bbox["x1"]); y1 = int(bbox["y1"])
                            crop = img[y0:y1, x0:x1]
                            if crop.size > 0:
                                fingers = count_fingers_mediapipe_from_image(crop)
                            else:
                                fingers = count_fingers_mediapipe_from_image(img)
                        except Exception:
                            fingers = count_fingers_mediapipe_from_image(img)
                    else:
                        fingers = count_fingers_mediapipe_from_image(img)

            except Exception as e:
                print(f"[SERVER] Erreur inference/parsing: {e}")
                traceback.print_exc()
                fingers = 0

            print(f"[SERVER] {device_id} frame {frame_index} (motion={motion_pixels}) => {fingers} doigts")

            try:
                if fingers == 5:
                    message = "5 doigts : C'est Yosser Msalmi"
                elif fingers == 4:
                    message = "4 doigts : C'est Moez Kessemtini"
                elif fingers == 3:
                    message = "3 doigts : C'est Yesmine Cherif"
                elif fingers == 2:
                    message = "2 doigts : C'est Taicir Cheikhrouhou"
                else:
                    message = f"{fingers} doigts détectés : inconnu"

                client_socket.send(message.encode('utf-8'))        
            except Exception as e:
                print(f"[SERVER] Erreur envoi réponse: {e}")
                break

    except Exception as e:
        print(f"[SERVER] Exception client {addr} (outer): {e}")
        traceback.print_exc()
    finally:
        try:
            client_socket.close()
        except:
            pass
        print(f"[SERVER] Déconnexion {addr}")

def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((SERVER_IP, SERVER_PORT))
    s.listen(16)
    print(f"[SERVER] Serveur prêt sur {SERVER_IP}:{SERVER_PORT}")

    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    try:
        while True:
            client_socket, addr = s.accept()
            executor.submit(handle_client, client_socket, addr)
    except KeyboardInterrupt:
        print("[SERVER] Arrêt demandé (KeyboardInterrupt)")
        s.close()
        executor.shutdown(wait=True)
        print("[SERVER] Terminé")
    finally:
        s.close()
        executor.shutdown(wait=True)
        print("[SERVER] Terminé")

if __name__ == "__main__":
    main()
