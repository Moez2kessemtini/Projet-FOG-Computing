# Hand Gesture Recognition over Network

Ce projet implémente un système de reconnaissance de gestes de la main (nombre de doigts levés) avec un **client** qui capture les images via une caméra et un **serveur** qui détecte les doigts et renvoie des messages personnalisés.  

Le client et le serveur communiquent via **sockets TCP**. Le client envoie des images toutes les 10 secondes, et le serveur renvoie un texte selon le nombre de doigts détectés.

---

## Fonctionnalités

- Capture vidéo sur le client avec **détection simple de mouvement** (ROI optionnel).  
- Compression adaptative des images en JPEG avant envoi.  
- Détection du nombre de doigts levés via **YOLOv8-pose** et **MediaPipe Hands** sur le serveur.  
- Réponse personnalisée envoyée au client selon le nombre de doigts levés.  
- Reconnexion automatique en cas de perte de connexion.

---

## Matériel requis
# Reconnaissance d’identité par détection du nombre de doigts levés dans une chaîne de Fog Computing

Ce projet implémente un système de **reconnaissance d’identité** basé sur le **nombre de doigts levés**, dans un contexte de **Fog Computing**.  

Le client capture des images via une caméra, effectue un pré-traitement simple (détection de mouvement, ROI optionnelle) et envoie les images au serveur via **sockets TCP**. Le serveur détecte le nombre de doigts levés et renvoie un message personnalisé indiquant l’identité correspondante.

---

## Fonctionnalités

- Capture vidéo sur le client avec **détection simple de mouvement** (ROI optionnel).  
- Compression adaptative des images en JPEG avant envoi pour réduire la bande passante.  
- Détection du nombre de doigts levés via **YOLOv8-pose** et **MediaPipe Hands** sur le serveur.  
- Réponse personnalisée envoyée au client selon le nombre de doigts levés.  
- Reconnexion automatique en cas de perte de connexion.

---

## Matériel requis

- PC Client avec caméra (Windows, Linux ou Mac).  
- PC Serveur capable d’exécuter Python 3 et d’utiliser YOLOv8.  

---

## Dépendances Python

Installer les packages nécessaires :

```bash
pip install opencv-python numpy ultralytics mediapipe

- PC Client avec caméra (Windows, Linux ou Mac).  
- PC Serveur capable d’exécuter Python 3 et d’utiliser YOLOv8.  

---

## Dépendances Python

Installer les packages nécessaires :

```bash
pip install opencv-python numpy ultralytics mediapipe
