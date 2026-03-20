from __future__ import annotations

import json
import math
import shutil
import sqlite3
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Iterable

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
IMAGE_DIR = DATA_DIR / "images"
DB_PATH = DATA_DIR / "catalog.db"
THUMBNAIL_SIZE = (320, 320)
IMAGE_EXTENSIONS = {".png", ".gif", ".ppm", ".pgm"}


def ensure_storage() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL,
                image_path TEXT NOT NULL,
                features TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


@dataclass
class Sample:
    label: str
    image_path: Path
    features: list[float]


class PhotoImageLoader:
    @staticmethod
    def load(path: Path) -> tk.PhotoImage:
        try:
            return tk.PhotoImage(file=str(path))
        except tk.TclError as exc:
            raise ValueError(
                "Formato não suportado nesta instalação do Tkinter. Use PNG, GIF, PPM ou PGM."
            ) from exc


class FeatureExtractor:
    """Extrai características sem dependências externas a partir do PhotoImage do Tkinter."""

    @staticmethod
    def extract(image_path: Path) -> list[float]:
        photo = PhotoImageLoader.load(image_path)
        width = photo.width()
        height = photo.height()
        if width <= 0 or height <= 0:
            raise ValueError("Imagem inválida.")

        resized = photo.subsample(max(1, width // 64), max(1, height // 64)) if width > 64 or height > 64 else photo
        pixels = []
        for y in range(resized.height()):
            row = []
            for x in range(resized.width()):
                color = resized.get(x, y)
                if isinstance(color, tuple):
                    rgb = color[:3]
                else:
                    rgb = FeatureExtractor._hex_to_rgb(color)
                row.append(rgb)
            pixels.append(row)

        dhash = FeatureExtractor._difference_hash(pixels)
        histogram = FeatureExtractor._color_histogram(pixels)
        edges = FeatureExtractor._edge_profile(pixels)
        return dhash + histogram + edges

    @staticmethod
    def _hex_to_rgb(color: str) -> tuple[int, int, int]:
        color = color.lstrip("#")
        if len(color) == 12:
            return tuple(int(color[idx:idx + 4], 16) // 257 for idx in range(0, 12, 4))  # type: ignore[return-value]
        if len(color) == 6:
            return tuple(int(color[idx:idx + 2], 16) for idx in range(0, 6, 2))  # type: ignore[return-value]
        raise ValueError(f"Cor inválida: {color}")

    @staticmethod
    def _to_grayscale(pixels: list[list[tuple[int, int, int]]]) -> list[list[float]]:
        return [[(r * 0.299 + g * 0.587 + b * 0.114) for r, g, b in row] for row in pixels]

    @staticmethod
    def _difference_hash(pixels: list[list[tuple[int, int, int]]]) -> list[float]:
        gray = FeatureExtractor._to_grayscale(pixels)
        rows = min(8, len(gray))
        cols = min(9, len(gray[0])) if gray else 0
        if rows < 1 or cols < 2:
            return [0.0] * 64

        step_y = max(1, len(gray) // rows)
        step_x = max(1, len(gray[0]) // cols)
        sampled = [[gray[y * step_y][x * step_x] for x in range(cols)] for y in range(rows)]

        bits: list[float] = []
        for row in sampled:
            for idx in range(cols - 1):
                bits.append(1.0 if row[idx] > row[idx + 1] else 0.0)
        while len(bits) < 64:
            bits.append(0.0)
        return bits[:64]

    @staticmethod
    def _color_histogram(pixels: list[list[tuple[int, int, int]]], bins: int = 8) -> list[float]:
        hist = [[0] * bins for _ in range(3)]
        total = 0
        for row in pixels:
            for rgb in row:
                total += 1
                for channel_idx, value in enumerate(rgb):
                    bucket = min(bins - 1, value * bins // 256)
                    hist[channel_idx][bucket] += 1
        total = float(total or 1)
        output: list[float] = []
        for channel in hist:
            output.extend(count / total for count in channel)
        return output

    @staticmethod
    def _edge_profile(pixels: list[list[tuple[int, int, int]]]) -> list[float]:
        gray = FeatureExtractor._to_grayscale(pixels)
        h_values = []
        v_values = []
        for row in gray:
            if len(row) > 1:
                h_values.append(sum(abs(row[idx] - row[idx + 1]) for idx in range(len(row) - 1)) / (len(row) - 1))
        for col in range(len(gray[0])):
            column = [gray[row][col] for row in range(len(gray))]
            if len(column) > 1:
                v_values.append(sum(abs(column[idx] - column[idx + 1]) for idx in range(len(column) - 1)) / (len(column) - 1))
        h_avg = sum(h_values) / len(h_values) / 255.0 if h_values else 0.0
        v_avg = sum(v_values) / len(v_values) / 255.0 if v_values else 0.0
        return [h_avg, v_avg]


class ImageRepository:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def add_sample(self, label: str, source_path: Path, features: list[float]) -> Path:
        label_slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_") or "classe"
        target_dir = IMAGE_DIR / label_slug
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / source_path.name
        counter = 1
        while target_path.exists():
            target_path = target_dir / f"{source_path.stem}_{counter}{source_path.suffix}"
            counter += 1
        shutil.copy2(source_path, target_path)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO samples (label, image_path, features) VALUES (?, ?, ?)",
                (label.strip(), str(target_path.relative_to(APP_DIR)), json.dumps(features)),
            )
            conn.commit()
        return target_path

    def list_samples(self) -> list[Sample]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT label, image_path, features FROM samples ORDER BY label, created_at DESC"
            ).fetchall()
        return [
            Sample(label=row[0], image_path=APP_DIR / row[1], features=json.loads(row[2]))
            for row in rows
        ]


class RecognitionService:
    def __init__(self, repository: ImageRepository) -> None:
        self.repository = repository

    def recognize(self, image_path: Path) -> dict[str, object] | None:
        samples = self.repository.list_samples()
        if not samples:
            return None

        query_features = FeatureExtractor.extract(image_path)
        scored: list[tuple[float, Sample]] = []
        for sample in samples:
            distance = self._euclidean_distance(query_features, sample.features)
            scored.append((distance, sample))

        scored.sort(key=lambda item: item[0])
        best_distance, best_sample = scored[0]
        confidence = self._distance_to_confidence(best_distance)
        alternatives = [
            {
                "label": sample.label,
                "distance": round(distance, 4),
                "confidence": round(self._distance_to_confidence(distance), 2),
            }
            for distance, sample in scored[:5]
        ]
        return {
            "label": best_sample.label,
            "distance": round(best_distance, 4),
            "confidence": round(confidence, 2),
            "sample_path": best_sample.image_path,
            "alternatives": alternatives,
        }

    @staticmethod
    def _euclidean_distance(first: Iterable[float], second: Iterable[float]) -> float:
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second, strict=True)))

    @staticmethod
    def _distance_to_confidence(distance: float) -> float:
        confidence = max(0.0, min(100.0, 100.0 * math.exp(-1.4 * distance)))
        return confidence


class ModernImageRecognitionApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        ensure_storage()
        self.title("Imagem Inteligente • Cadastro e Reconhecimento")
        self.geometry("1180x720")
        self.minsize(980, 640)
        self.configure(bg="#0f172a")

        self.repository = ImageRepository(DB_PATH)
        self.recognition_service = RecognitionService(self.repository)
        self.selected_train_images: list[Path] = []
        self.selected_recognition_image: Path | None = None
        self._preview_photo: tk.PhotoImage | None = None

        self._setup_style()
        self._build_layout()
        self.refresh_dataset_summary()

    def _setup_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#0f172a")
        style.configure("Card.TFrame", background="#111827")
        style.configure("TLabel", background="#111827", foreground="#e5e7eb", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI Semibold", 22), foreground="#f8fafc", background="#0f172a")
        style.configure("Subtitle.TLabel", font=("Segoe UI", 11), foreground="#94a3b8", background="#0f172a")
        style.configure("Section.TLabel", font=("Segoe UI Semibold", 14), foreground="#f8fafc", background="#111827")
        style.configure("Accent.TButton", font=("Segoe UI Semibold", 10), padding=10)
        style.map("Accent.TButton", background=[("active", "#2563eb")])
        style.configure("Treeview", background="#0b1220", fieldbackground="#0b1220", foreground="#e5e7eb", rowheight=28)
        style.configure("Treeview.Heading", background="#1f2937", foreground="#f8fafc", font=("Segoe UI Semibold", 10))
        style.configure("TEntry", fieldbackground="#0b1220", foreground="#f8fafc")

    def _build_layout(self) -> None:
        root = ttk.Frame(self)
        root.pack(fill="both", expand=True, padx=20, pady=20)
        root.columnconfigure(0, weight=3)
        root.columnconfigure(1, weight=2)
        root.rowconfigure(1, weight=1)

        header = ttk.Frame(root)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 16))
        ttk.Label(header, text="Sistema de Reconhecimento de Imagens", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Cadastre imagens por categoria, treine a base localmente e identifique novas imagens com uma interface moderna.",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(6, 0))

        left_panel = ttk.Frame(root, style="Card.TFrame", padding=16)
        left_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        left_panel.columnconfigure(0, weight=1)

        right_panel = ttk.Frame(root, style="Card.TFrame", padding=16)
        right_panel.grid(row=1, column=1, sticky="nsew")
        right_panel.columnconfigure(0, weight=1)

        self._build_training_section(left_panel)
        self._build_recognition_section(right_panel)

    def _build_training_section(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="1. Cadastro para treinamento", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            parent,
            text="Selecione imagens semelhantes e informe o nome da classe (ex.: maçã, caneca, logotipo).",
        ).grid(row=1, column=0, sticky="w", pady=(6, 12))

        form = ttk.Frame(parent, style="Card.TFrame")
        form.grid(row=2, column=0, sticky="ew")
        form.columnconfigure(1, weight=1)

        ttk.Label(form, text="Nome da categoria:").grid(row=0, column=0, sticky="w")
        self.label_entry = ttk.Entry(form)
        self.label_entry.grid(row=0, column=1, sticky="ew", padx=(12, 0))

        actions = ttk.Frame(parent, style="Card.TFrame")
        actions.grid(row=3, column=0, sticky="ew", pady=(12, 12))
        ttk.Button(actions, text="Selecionar imagens", style="Accent.TButton", command=self.select_training_images).pack(side="left")
        ttk.Button(actions, text="Cadastrar no sistema", command=self.save_training_samples).pack(side="left", padx=(10, 0))
        ttk.Button(actions, text="Atualizar lista", command=self.refresh_dataset_summary).pack(side="left", padx=(10, 0))

        self.selected_files_label = ttk.Label(parent, text="Nenhuma imagem selecionada.")
        self.selected_files_label.grid(row=4, column=0, sticky="w", pady=(0, 12))

        columns = ("categoria", "arquivo")
        self.dataset_tree = ttk.Treeview(parent, columns=columns, show="headings", height=12)
        self.dataset_tree.heading("categoria", text="Categoria")
        self.dataset_tree.heading("arquivo", text="Arquivo")
        self.dataset_tree.column("categoria", width=160, anchor="w")
        self.dataset_tree.column("arquivo", width=420, anchor="w")
        self.dataset_tree.grid(row=5, column=0, sticky="nsew")
        parent.rowconfigure(5, weight=1)

    def _build_recognition_section(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="2. Reconhecimento", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            parent,
            text="Escolha uma imagem e compare com o cadastro salvo localmente.",
        ).grid(row=1, column=0, sticky="w", pady=(6, 12))

        actions = ttk.Frame(parent, style="Card.TFrame")
        actions.grid(row=2, column=0, sticky="ew")
        ttk.Button(actions, text="Escolher imagem", style="Accent.TButton", command=self.select_recognition_image).pack(side="left")
        ttk.Button(actions, text="Reconhecer agora", command=self.run_recognition).pack(side="left", padx=(10, 0))

        self.preview_label = ttk.Label(parent, text="A pré-visualização da imagem aparecerá aqui.", anchor="center")
        self.preview_label.grid(row=3, column=0, sticky="nsew", pady=(16, 12))
        parent.rowconfigure(3, weight=1)

        self.result_text = tk.Text(
            parent,
            height=12,
            bg="#0b1220",
            fg="#e5e7eb",
            insertbackground="#e5e7eb",
            relief="flat",
            font=("Consolas", 10),
            wrap="word",
        )
        self.result_text.grid(row=4, column=0, sticky="ew")
        self.result_text.insert(
            "1.0",
            "Sugestão de uso:\n"
            "• Cadastre entre 5 e 20 imagens por categoria.\n"
            "• Misture ângulos, iluminação e fundos diferentes.\n"
            "• Para produção futura, você pode evoluir para IA com embeddings (CLIP / TensorFlow / PyTorch).",
        )
        self.result_text.configure(state="disabled")

    def select_training_images(self) -> None:
        files = filedialog.askopenfilenames(
            title="Selecione as imagens de treinamento",
            filetypes=[("Imagens suportadas pelo Tkinter", "*.png *.gif *.ppm *.pgm")],
        )
        self.selected_train_images = [Path(path) for path in files if Path(path).suffix.lower() in IMAGE_EXTENSIONS]
        if self.selected_train_images:
            names = ", ".join(path.name for path in self.selected_train_images[:3])
            extra = "" if len(self.selected_train_images) <= 3 else f" ... (+{len(self.selected_train_images) - 3})"
            self.selected_files_label.config(text=f"{len(self.selected_train_images)} imagem(ns): {names}{extra}")
        else:
            self.selected_files_label.config(text="Nenhuma imagem selecionada.")

    def save_training_samples(self) -> None:
        label = self.label_entry.get().strip()
        if not label:
            messagebox.showwarning("Categoria obrigatória", "Informe o nome da categoria antes de cadastrar.")
            return
        if not self.selected_train_images:
            messagebox.showwarning("Imagens obrigatórias", "Selecione uma ou mais imagens para treinamento.")
            return

        saved_count = 0
        for image_path in self.selected_train_images:
            try:
                features = FeatureExtractor.extract(image_path)
                self.repository.add_sample(label, image_path, features)
                saved_count += 1
            except ValueError as exc:
                messagebox.showwarning("Imagem ignorada", f"{image_path.name}: {exc}")

        self.label_entry.delete(0, tk.END)
        self.selected_train_images = []
        self.selected_files_label.config(text="Nenhuma imagem selecionada.")
        self.refresh_dataset_summary()
        messagebox.showinfo("Treinamento atualizado", f"{saved_count} imagem(ns) cadastrada(s) na categoria '{label}'.")

    def refresh_dataset_summary(self) -> None:
        for item in self.dataset_tree.get_children():
            self.dataset_tree.delete(item)
        for sample in self.repository.list_samples():
            self.dataset_tree.insert("", "end", values=(sample.label, sample.image_path.name))

    def select_recognition_image(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Escolha uma imagem para reconhecimento",
            filetypes=[("Imagens suportadas pelo Tkinter", "*.png *.gif *.ppm *.pgm")],
        )
        if not file_path:
            return
        self.selected_recognition_image = Path(file_path)
        self._load_preview(self.selected_recognition_image)

    def _load_preview(self, image_path: Path) -> None:
        try:
            photo = PhotoImageLoader.load(image_path)
        except ValueError as exc:
            messagebox.showwarning("Formato não suportado", str(exc))
            return
        divisor_x = max(1, photo.width() // THUMBNAIL_SIZE[0])
        divisor_y = max(1, photo.height() // THUMBNAIL_SIZE[1])
        self._preview_photo = photo.subsample(divisor_x, divisor_y) if divisor_x > 1 or divisor_y > 1 else photo
        self.preview_label.configure(image=self._preview_photo, text="")

    def run_recognition(self) -> None:
        if not self.selected_recognition_image:
            messagebox.showwarning("Imagem obrigatória", "Escolha uma imagem para reconhecimento.")
            return

        try:
            result = self.recognition_service.recognize(self.selected_recognition_image)
        except ValueError as exc:
            messagebox.showwarning("Erro no reconhecimento", str(exc))
            return

        if result is None:
            messagebox.showwarning("Base vazia", "Cadastre imagens de treinamento antes de realizar o reconhecimento.")
            return

        confidence = result["confidence"]
        decision = "Correspondência forte" if confidence >= 70 else "Correspondência aproximada"
        alternative_lines = [
            f"- {item['label']}: confiança {item['confidence']}% | distância {item['distance']}"
            for item in result["alternatives"]
        ]
        output = (
            f"Resultado principal: {result['label']}\n"
            f"Nível de confiança: {confidence}%\n"
            f"Distância vetorial: {result['distance']}\n"
            f"Status: {decision}\n\n"
            "Top correspondências:\n"
            + "\n".join(alternative_lines)
        )
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert("1.0", output)
        self.result_text.configure(state="disabled")


def main() -> None:
    app = ModernImageRecognitionApp()
    app.mainloop()


if __name__ == "__main__":
    main()
