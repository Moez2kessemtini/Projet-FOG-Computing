# 🤚 Reconnaissance d’identité par détection du nombre de doigts levés dans une chaîne de Fog Computing

Ce projet est un **système de reconnaissance de personne via les gestes de la main** (nombre de doigts levés) conçu dans le cadre d'un projet académique de **Fog Computing**.  
L’objectif est de simplifier la reconnaissance d’identité, souvent complexe via la reconnaissance faciale, en utilisant le **nombre de doigts levés** comme signal d’identification.

Le projet exploite pleinement les principes du **Fog Computing**, avec des **nœuds intermédiaires** qui traitent les images progressivement avant de les transmettre au serveur principal pour détection avancée.

---

## 🏗️ Architecture et flux Fog Computing

Le système est organisé sur une **chaîne de nœuds Fog** :  

1. **Caméra** : capture les images et transmet l'image vers le PC hote 
2. **Nœuds Fog intermédiaire** : C'est le noeud intérmédiaire qui reçoit les images, applique des traitements légers (compression, filtrage, réduction de taille) et les transmet au nœud suivant, optimisant la bande passante et réduisant la latence.  
3. **Serveur principal** : reçoit les images finales, effectue la détection avancée du nombre de doigts levés via **YOLOv8-pose** et **MediaPipe Hands**, puis renvoie un message personnalisé indiquant l’identité du personne en se basant sur le nombre de doigts détectés. 

**Transmission** : les images sont envoyées via **sockets TCP**, avec un envoi toutes les 2 secondes pour un traitement en temps quasi réel.


---

## ✨ Fonctionnalités

- 🎥 Capture vidéo par le caméra.
- 🖧 Traitement préliminaire pour **détection simple de mouvement** (ROI optionnel).  
- 🖼️ Compression adaptative des images en JPEG avant envoi pour réduire la bande passante.  
- 🖧 Transmission via une **chaîne de nœuds Fog** pour un traitement progressif et distribué.  
- 🤖 Détection du nombre de doigts levés sur le serveur principal pour une **reconnaissance d’identité simplifiée**.  
- 💬 Réponse personnalisée envoyée au client selon le nombre de doigts détectés.  
- 🔄 Reconnexion automatique en cas de perte de connexion.  

---

## 🖥️ Matériel requis

- PC Client avec caméra (Windows, Linux ou Mac).  
- PC Serveur capable d’exécuter Python 3 et d’utiliser YOLOv8.  

---

## 🐍 Dépendances Python

Installer les packages nécessaires :  

**pip install opencv-python numpy ultralytics mediapipe**

---

# 🚀 Usage

## 1️⃣ Lancer le serveur principal :

``` bash
python server.py
```

## 2️⃣ Lancer le client :

``` bash
python client.py
```

Le client envoie les images toutes les 2 secondes via la chaîne de
nœuds Fog.\
Le serveur détecte le nombre de doigts levés et renvoie un message
correspondant à l'identité associée.
