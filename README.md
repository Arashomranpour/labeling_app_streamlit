<div align="center">

# 🏷️ Labeling App with Streamlit

**Automatically label text, YouTube videos and CSV data with topic modeling - in a simple web interface.**

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Whisper](https://img.shields.io/badge/OpenAI_Whisper-412991?logo=openai&logoColor=white)
![Gensim](https://img.shields.io/badge/Gensim-LDA-informational)

</div>

---

![App screenshot](https://github.com/user-attachments/assets/eacb2bdd-a5eb-458d-ab54-e51499d6e05b)
![App screenshot](https://github.com/user-attachments/assets/350dbf39-369b-46cf-8c2c-050ffbcd7978)

## ✨ Features

Choose a labeling mode from the sidebar:

| Mode | What it does |
|---|---|
| 📝 **On Text** | Paste text, run **LDA topic modeling** (Gensim) and get topic labels based on a built-in insurance keyword list |
| 🎬 **On Video** | Enter a YouTube URL → download the audio → transcribe it with **OpenAI Whisper** → label the transcript |
| 📊 **On CSV** | Upload a CSV file and analyze its text content |

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- [FFmpeg](https://ffmpeg.org/) installed (required by Whisper)

### Install & run

```bash
git clone https://github.com/Arashomranpour/labeling_app_streamlit.git
cd labeling_app_streamlit
pip install -r requirements.txt
streamlit run app.py
```

> ℹ️ `heapq` in `requirements.txt` is part of the Python standard library and does not need to be installed separately.

## 📁 Project Structure

```
.
├── app.py              # Streamlit UI, topic modeling and transcription
├── requirements.txt
└── README.md
```

## 🗺️ Roadmap

- Support more file formats and dataset types
- Multi-user labeling
- Customizable label sets

## 🛠️ Tech Stack

`Streamlit` · `Gensim` · `OpenAI Whisper` · `pytube` · `pandas` · `NumPy`

## 🤝 Contributing

Contributions are welcome - fork the repo, open an issue or submit a pull request.
