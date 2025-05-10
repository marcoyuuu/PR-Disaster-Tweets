# 🧠 PR-Disaster-Tweets-RoBERTa

Análisis de sentimiento de tweets en desastres naturales usando el modelo `cardiffnlp/twitter-roberta-base-sentiment-latest` con aceleración por GPU (CUDA 12.1).  
Este proyecto está optimizado para entornos con **NVIDIA RTX 4070** y **Python 3.11+** en Windows 11.

---

## ⚙️ Requisitos

- Python 3.11+
- GPU NVIDIA con soporte CUDA 12.1 (ej. RTX 4070)
- VS Code recomendado

---

## 🚀 Instalación Paso a Paso

### 1. Clona el repositorio (opcional)

```bash
git clone https://github.com/tu-usuario/PR-Disaster-Tweets-RoBERTa.git
cd PR-Disaster-Tweets-RoBERTa
```

Buena observación. En términos generales:

- ✅ **Si tu entorno tiene bien configurado `python` o `py` en el PATH**, entonces basta con:

  ```bash
  python -m venv .venv
  ```

  o

  ```bash
  py -3.11 -m venv .venv
  ```

- ❌ No es necesario (ni recomendable en un README público) usar rutas absolutas como:

  ```bash
  C:\Users\Marco\AppData\Local\Programs\Python\Python311\python.exe -m venv .venv
  ```

  …porque **eso es específico de tu máquina personal**, y rompe portabilidad.

---

### 2. Crea el entorno virtual

```bash
python -m venv .venv
```

> Si tienes varias versiones de Python instaladas, puedes usar:

```bash
py -3.11 -m venv .venv
```

### 3. Activa el entorno virtual (PowerShell)

```powershell
.venv\Scripts\Activate
```

> Si ves un error sobre ejecución de scripts, ejecuta esto una vez:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

### 4. Instala las dependencias

```powershell
pip install -r requirements.txt
```

Este archivo instala:

- `transformers` y `torch` con soporte para CUDA 12.1
- `pandas`, `emoji`, `scipy`, `matplotlib`, `seaborn`, etc.

---

## 📂 Estructura del Proyecto

```
PR-Disaster-Tweets-RoBERTa/
├── data/
│   └── tweets.csv               # CSV con columna 'full_text' o similar
├── results/
│   └── sentiment_results.csv    # Resultados generados automáticamente
├── main.py                      # Script principal de análisis
├── requirements.txt
└── README.md
```

---

## 🧪 Ejecutar análisis de sentimiento

1. Asegúrate de tener un archivo CSV en la carpeta `data/` con una columna llamada `full_text`, `tweet`, `clean_text` o `text`.

2. Corre el script principal (ahora puedes especificar el archivo de entrada y salida):

```bash
python main.py --input_csv data/tweets.csv --output_csv results/sentiment_results.csv
```

Si no especificas los argumentos, usará por defecto:

- Entrada: `data/HumAID_maria_tweets_clean.csv`
- Salida: `results/sentiment_results.csv`

3. Al finalizar, los resultados aparecerán en el archivo de salida que indiques.

Con dos columnas nuevas:

- `sentiment`: etiqueta (Positive, Neutral, Negative)
- `confidence`: nivel de certeza del modelo

---

## 🧠 Modelo utilizado

- [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
- Entrenado para entender emojis, texto informal y lenguaje tipo Twitter

---

## ✅ Autor y licencia

Proyecto desarrollado por Marco para propósitos educativos e investigativos.
Licencia MIT. Puedes modificar y reutilizar este proyecto libremente.
