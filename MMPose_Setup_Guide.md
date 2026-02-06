# MMPose – Vollständiger Setup-Guide (Ubuntu 22.04 + Docker)

> **Ziel:** MMPose (v1.3.x) als Docker-Container auf Ubuntu 22.04 aufsetzen und erste Inferenz durchführen.
> **Stand:** Februar 2026 | Getestet für Ubuntu 22.04 LTS

---

## Inhaltsverzeichnis

1. [Kompatibilität: Ubuntu 22.04](#1-kompatibilität-ubuntu-2204)
2. [Voraussetzungen prüfen](#2-voraussetzungen-prüfen)
3. [Docker installieren (falls nicht vorhanden)](#3-docker-installieren)
4. [NVIDIA Container Toolkit installieren](#4-nvidia-container-toolkit-installieren)
5. [MMPose-Repository klonen](#5-mmpose-repository-klonen)
6. [Docker-Image bauen (offizieller Weg)](#6-docker-image-bauen-offizieller-weg)
7. [Alternativer Weg: Eigenes Dockerfile (empfohlen)](#7-alternativer-weg-eigenes-dockerfile-empfohlen)
8. [Container starten](#8-container-starten)
9. [Installation verifizieren](#9-installation-verifizieren)
10. [Erste Inferenz durchführen](#10-erste-inferenz-durchführen)
11. [Nützliche Befehle & Tipps](#11-nützliche-befehle--tipps)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Kompatibilität: Ubuntu 22.04

**Ja, MMPose läuft problemlos auf Ubuntu 22.04.** MMPose unterstützt Linux offiziell und hat folgende Mindestanforderungen:

| Anforderung       | Minimum        | Empfohlen            |
|--------------------|----------------|----------------------|
| Betriebssystem     | Linux (x86_64) | Ubuntu 20.04 / 22.04 |
| Python             | 3.7+           | 3.8 – 3.10           |
| PyTorch            | 1.8+           | 2.0+                 |
| CUDA               | 9.2+           | 11.7+ / 12.x         |
| Docker             | 19.03+         | 24.x+                |

Bei der Docker-Variante brauchst du dir um Python-Version und CUDA-Version auf dem Host-System keine Sorgen zu machen – alles wird im Container verwaltet.

---

## 2. Voraussetzungen prüfen

Führe folgende Befehle aus, um den aktuellen Stand deines Systems zu prüfen:

### 2.1 GPU und NVIDIA-Treiber

```bash
# Prüfe ob NVIDIA-Treiber installiert sind
nvidia-smi
```

Du solltest eine Tabelle mit GPU-Name, Treiberversion und CUDA-Version sehen. Falls nicht, installiere zuerst die NVIDIA-Treiber:

```bash
sudo apt update
sudo ubuntu-drivers autoinstall
sudo reboot
```

### 2.2 Docker prüfen

```bash
# Prüfe ob Docker installiert ist
docker --version

# Prüfe ob Docker-Daemon läuft
sudo systemctl status docker
```

### 2.3 NVIDIA Container Toolkit prüfen

```bash
# Prüfe ob GPU im Container sichtbar ist
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

Wenn einer dieser Schritte fehlschlägt, folge den entsprechenden Installationsabschnitten unten.

---

## 3. Docker installieren

Falls Docker noch nicht installiert ist:

```bash
# Alte Versionen entfernen (falls vorhanden)
sudo apt remove docker docker-engine docker.io containerd runc 2>/dev/null

# Abhängigkeiten installieren
sudo apt update
sudo apt install -y ca-certificates curl gnupg lsb-release

# Docker GPG-Schlüssel hinzufügen
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Docker-Repository hinzufügen
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Docker installieren
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Deinen Benutzer zur Docker-Gruppe hinzufügen (kein sudo nötig)
sudo usermod -aG docker $USER
newgrp docker

# Testen
docker run hello-world
```

---

## 4. NVIDIA Container Toolkit installieren

Das NVIDIA Container Toolkit ist **zwingend erforderlich**, damit der Docker-Container auf die GPU zugreifen kann.

```bash
# NVIDIA Container Toolkit Repository einrichten
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Installieren
sudo apt update
sudo apt install -y nvidia-container-toolkit

# Docker-Runtime konfigurieren
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Testen – sollte GPU-Infos anzeigen
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

---

## 5. MMPose-Repository klonen

```bash
cd ~
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
```

---

## 6. Docker-Image bauen (offizieller Weg)

MMPose stellt im Repository unter `docker/` ein Dockerfile bereit. Dieses basiert allerdings auf älteren CUDA/PyTorch-Versionen und muss ggf. angepasst werden.

```bash
# Im mmpose-Verzeichnis:
docker build -t mmpose docker/
```

> **Hinweis:** Das offizielle Dockerfile nutzt ältere Versionen (PyTorch 1.8, CUDA 10.1). Für modernere GPUs (RTX 30xx/40xx) wird der alternative Weg in Abschnitt 7 empfohlen.

---

## 7. Alternativer Weg: Eigenes Dockerfile (empfohlen)

Erstelle ein aktualisiertes Dockerfile, das mit modernen GPUs und Ubuntu 22.04 funktioniert:

```bash
# Erstelle das Dockerfile
mkdir -p ~/mmpose-docker
cat > ~/mmpose-docker/Dockerfile << 'EOF'
# ============================================================
# MMPose v1.3.x – Docker-Image für Ubuntu 22.04 + CUDA 11.8
# ============================================================
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Umgebungsvariablen
ENV DEBIAN_FRONTEND=noninteractive
ENV FORCE_CUDA="1"
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6 8.9 9.0+PTX"

# System-Abhängigkeiten
RUN apt-get update && apt-get install -y \
    git wget curl \
    python3 python3-pip python3-dev \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python-Symlink und pip upgrade
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    pip install --upgrade pip setuptools wheel

# PyTorch mit CUDA 11.8 installieren
RUN pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# OpenMMLab-Abhängigkeiten installieren
RUN pip install -U openmim && \
    mim install mmengine && \
    mim install "mmcv>=2.0.1" && \
    mim install "mmdet>=3.1.0"

# MMPose von Source installieren
RUN git clone https://github.com/open-mmlab/mmpose.git /mmpose
WORKDIR /mmpose
RUN pip install -r requirements.txt && \
    pip install -v -e .

# Daten-Verzeichnis erstellen
RUN mkdir -p /mmpose/data

# Standard-Einstiegspunkt
CMD ["/bin/bash"]
EOF
```

### Image bauen

```bash
cd ~/mmpose-docker
docker build -t mmpose:latest .
```

Der Build-Prozess dauert je nach Internetverbindung und Hardware **15–30 Minuten**.

---

## 8. Container starten

### 8.1 Interaktive Shell

```bash
docker run --gpus all \
    --shm-size=8g \
    -it \
    -v ~/mmpose-data:/mmpose/data \
    -v ~/mmpose-output:/mmpose/output \
    mmpose:latest
```

### 8.2 Parameter erklärt

| Parameter          | Bedeutung                                                     |
|--------------------|---------------------------------------------------------------|
| `--gpus all`       | Alle GPUs dem Container zur Verfügung stellen                 |
| `--shm-size=8g`    | Shared Memory erhöhen (nötig für DataLoader mit num_workers)  |
| `-it`              | Interaktiver Modus mit Terminal                               |
| `-v ~/mmpose-data:/mmpose/data` | Lokales Datenverzeichnis in den Container mounten |
| `-v ~/mmpose-output:/mmpose/output` | Ausgabeverzeichnis mounten                  |

### 8.3 Container mit Port-Weiterleitung (z.B. für Jupyter)

```bash
docker run --gpus all \
    --shm-size=8g \
    -it \
    -p 8888:8888 \
    -v ~/mmpose-data:/mmpose/data \
    mmpose:latest \
    bash -c "pip install jupyter && jupyter notebook --ip=0.0.0.0 --allow-root --no-browser"
```

---

## 9. Installation verifizieren

Führe die folgenden Befehle **innerhalb des Containers** aus:

```bash
# Python-Version prüfen
python --version

# PyTorch und CUDA prüfen
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA verfügbar: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# MMPose-Version prüfen
python -c "import mmpose; print(f'MMPose: {mmpose.__version__}')"

# MMCV prüfen
python -c "import mmcv; print(f'MMCV: {mmcv.__version__}')"

# MMEngine prüfen
python -c "import mmengine; print(f'MMEngine: {mmengine.__version__}')"
```

Erwartete Ausgabe (ähnlich):

```
PyTorch: 2.1.0
CUDA verfügbar: True
GPU: NVIDIA GeForce RTX ...
MMPose: 1.3.2
MMCV: 2.1.0
MMEngine: 0.10.x
```

---

## 10. Erste Inferenz durchführen

### 10.1 Modell und Config herunterladen

```bash
# Innerhalb des Containers
cd /mmpose
mim download mmpose \
    --config td-hm_hrnet-w48_8xb32-210e_coco-256x192 \
    --dest .
```

### 10.2 Inferenz auf einem Testbild

```bash
python demo/image_demo.py \
    tests/data/coco/000000000785.jpg \
    td-hm_hrnet-w48_8xb32-210e_coco-256x192.py \
    td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth \
    --out-file output/vis_results.jpg \
    --draw-heatmap
```

### 10.3 Mit dem MMPoseInferencer (einfacher)

```python
# In einer Python-Shell oder als Script
from mmpose.apis import MMPoseInferencer

inferencer = MMPoseInferencer('human')
result_generator = inferencer(
    'tests/data/coco/000000000785.jpg',
    show=False,
    out_dir='output/'
)
result = next(result_generator)
print("Inferenz abgeschlossen! Ergebnis gespeichert in output/")
```

### 10.4 RTMPose (schnelles Real-Time-Modell)

```bash
# RTMPose für schnelle Echtzeit-Posenerkennung
python demo/image_demo.py \
    tests/data/coco/000000000785.jpg \
    projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth \
    --out-file output/rtmpose_result.jpg
```

---

## 11. Nützliche Befehle & Tipps

### Container-Management

```bash
# Laufende Container anzeigen
docker ps

# Gestoppte Container anzeigen
docker ps -a

# Container erneut starten und verbinden
docker start <CONTAINER_ID>
docker attach <CONTAINER_ID>

# Container mit neuem Terminal öffnen
docker exec -it <CONTAINER_ID> bash

# Container löschen
docker rm <CONTAINER_ID>

# Image löschen
docker rmi mmpose:latest
```

### Daten in den Container kopieren

```bash
# Vom Host in den Container
docker cp /pfad/zu/datei <CONTAINER_ID>:/mmpose/data/

# Vom Container zum Host
docker cp <CONTAINER_ID>:/mmpose/output/vis_results.jpg ~/
```

### docker-compose.yml (optional)

Erstelle eine `docker-compose.yml` für einfacheres Starten:

```yaml
version: '3.8'

services:
  mmpose:
    image: mmpose:latest
    runtime: nvidia
    shm_size: '8g'
    volumes:
      - ~/mmpose-data:/mmpose/data
      - ~/mmpose-output:/mmpose/output
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    stdin_open: true
    tty: true
```

Starten mit:

```bash
docker compose up -d
docker compose exec mmpose bash
```

---

## 12. Troubleshooting

### Problem: `docker: Error response from daemon: could not select device driver`

**Ursache:** NVIDIA Container Toolkit ist nicht installiert oder nicht konfiguriert.

**Lösung:** Abschnitt 4 (NVIDIA Container Toolkit) nochmals durchführen.

### Problem: `RuntimeError: CUDA out of memory`

**Lösung:** Batch-Size in der Config reduzieren oder `--shm-size` erhöhen:

```bash
docker run --gpus all --shm-size=16g -it mmpose:latest
```

### Problem: `ImportError: libGL.so.1: cannot open shared object file`

**Lösung:** Im Container:

```bash
apt-get update && apt-get install -y libgl1-mesa-glx
```

(Im empfohlenen Dockerfile oben ist das bereits enthalten.)

### Problem: Docker-Build schlägt bei `mim install mmcv` fehl

**Lösung:** MMCV-Build benötigt viel RAM. Erhöhe Docker-Speicher oder installiere MMCV als pre-built wheel:

```bash
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
```

### Problem: `permission denied` bei Docker-Befehlen

**Lösung:**

```bash
sudo usermod -aG docker $USER
# Danach ausloggen und neu einloggen, oder:
newgrp docker
```

### Problem: Alte GPU (Compute Capability < 6.0)

Passe `TORCH_CUDA_ARCH_LIST` im Dockerfile an deine GPU an. Prüfe die Compute Capability deiner GPU unter: https://developer.nvidia.com/cuda-gpus

---

## Kurzreferenz – Kompletter Workflow

```bash
# 1. Repo klonen
git clone https://github.com/open-mmlab/mmpose.git && cd mmpose

# 2. Docker-Image bauen (mit dem angepassten Dockerfile aus Abschnitt 7)
docker build -t mmpose:latest -f ~/mmpose-docker/Dockerfile .

# 3. Container starten
docker run --gpus all --shm-size=8g -it \
    -v $(pwd)/data:/mmpose/data \
    -v $(pwd)/output:/mmpose/output \
    mmpose:latest

# 4. Im Container: Verifizieren
python -c "import mmpose; print(mmpose.__version__)"

# 5. Im Container: Inferenz testen
python demo/image_demo.py tests/data/coco/000000000785.jpg \
    td-hm_hrnet-w48_8xb32-210e_coco-256x192.py \
    td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth \
    --out-file output/vis_results.jpg --draw-heatmap
```

---

*Erstellt: Februar 2026 | Quelle: [MMPose Docs](https://mmpose.readthedocs.io/en/latest/installation.html) & [GitHub](https://github.com/open-mmlab/mmpose)*
