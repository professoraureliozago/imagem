from __future__ import annotations

import importlib.util
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
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".ppm", ".pgm"}
SUPPORTED_FILE_TYPES = [("Imagens", "*.png *.jpg *.jpeg *.bmp *.gif *.webp *.ppm *.pgm")]
CLASSIC_BACKEND = "classic"
AI_BACKEND = "ai_clip"


@dataclass
class BackendStatus:
    requested_backend: str
    active_backend: str
    description: str
    ready: bool


@dataclass
class Sample:
    label: str
    image_path: Path
    features: list[float]
    backend: str


class DependencyManager:
    @staticmethod
    def has_pillow() -> bool:
        return importlib.util.find_spec("PIL") is not None

    @staticmethod
    def has_ai_stack() -> bool:
        return (
            importlib.util.find_spec("torch") is not None
            and importlib.util.find_spec("transformers") is not None
        )

    @staticmethod
    def ensure_pillow() -> None:
        if not DependencyManager.has_pillow():
            raise RuntimeError(
                "Este aplicativo precisa da biblioteca Pillow para abrir imagens comuns. "
                "Instale com: pip install -r requirements.txt"
            )

    @staticmethod
    def import_pillow():
        DependencyManager.ensure_pillow()
        from PIL import Image, ImageOps, ImageTk

        return Image, ImageOps, ImageTk

    @staticmethod
    def import_ai_stack():
        if not DependencyManager.has_ai_stack():
            raise RuntimeError(
                "O modo IA precisa de torch e transformers. "
                "Instale com: pip install -r requirements.txt"
            )
        import torch
        from transformers import CLIPModel, CLIPProcessor

        return torch, CLIPModel, CLIPProcessor


class BackendManager:
    @staticmethod
    def get_status(requested_backend: str) -> BackendStatus:
        if requested_backend == AI_BACKEND:
            if DependencyManager.has_ai_stack():
                return BackendStatus(
                    requested_backend=AI_BACKEND,
                    active_backend=AI_BACKEND,
                    description="IA ativa com embeddings CLIP.",
                    ready=True,
                )
            return BackendStatus(
                requested_backend=AI_BACKEND,
                active_backend=CLASSIC_BACKEND,
                description="Modo IA indisponível no ambiente. Usando fallback clássico.",
                ready=False,
            )
        return BackendStatus(
            requested_backend=CLASSIC_BACKEND,
            active_backend=CLASSIC_BACKEND,
            description="Modo clássico ativo com hash perceptual + histograma + bordas.",
            ready=True,
        )


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
                backend TEXT NOT NULL DEFAULT 'classic',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        columns = {row[1] for row in conn.execute("PRAGMA table_info(samples)")}
        if "backend" not in columns:
            conn.execute("ALTER TABLE samples ADD COLUMN backend TEXT NOT NULL DEFAULT 'classic'")
        conn.commit()


class PillowImageLoader:
    @staticmethod
    def load_rgb(path: Path):
        Image, ImageOps, _ = DependencyManager.import_pillow()
        try:
            with Image.open(path) as image:
                return ImageOps.exif_transpose(image).convert("RGB")
        except OSError as exc:
            raise ValueError(f"Não foi possível abrir a imagem '{path.name}'.") from exc

    @staticmethod
    def load_preview(path: Path):
        Image, _, ImageTk = DependencyManager.import_pillow()
        image = PillowImageLoader.load_rgb(path)
        image.thumbnail(THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(image)


class ClassicFeatureExtractor:
    @staticmethod
    def extract(image_path: Path) -> list[float]:
        Image, _, _ = DependencyManager.import_pillow()
        image = PillowImageLoader.load_rgb(image_path)
        if image.width <= 0 or image.height <= 0:
            raise ValueError("Imagem inválida.")

        resized = image.copy()
        resized.thumbnail((64, 64), Image.Resampling.LANCZOS)
        pixel_access = resized.load()
        rows = [
            [pixel_access[x, y] for x in range(resized.width)]
            for y in range(resized.height)
        ]

        dhash = ClassicFeatureExtractor._difference_hash(rows)
        histogram = ClassicFeatureExtractor._color_histogram(rows)
        edges = ClassicFeatureExtractor._edge_profile(rows)
        return dhash + histogram + edges

    @staticmethod
    def _to_grayscale(pixels: list[list[tuple[int, int, int]]]) -> list[list[float]]:
        return [[(r * 0.299 + g * 0.587 + b * 0.114) for r, g, b in row] for row in pixels]

    @staticmethod
    def _difference_hash(pixels: list[list[tuple[int, int, int]]]) -> list[float]:
        gray = ClassicFeatureExtractor._to_grayscale(pixels)
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
        gray = ClassicFeatureExtractor._to_grayscale(pixels)
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


class AIEmbeddingExtractor:
    MODEL_NAME = "openai/clip-vit-base-patch32"
    _model = None
    _processor = None
    _device = None

    @classmethod
    def _bootstrap(cls) -> None:
        if cls._model is not None and cls._processor is not None and cls._device is not None:
            return
        torch, CLIPModel, CLIPProcessor = DependencyManager.import_ai_stack()
        cls._device = "cuda" if torch.cuda.is_available() else "cpu"
        cls._processor = CLIPProcessor.from_pretrained(cls.MODEL_NAME)
        cls._model = CLIPModel.from_pretrained(cls.MODEL_NAME).to(cls._device)
        cls._model.eval()

    @classmethod
    def _project_if_needed(cls, pooled, torch):
        projection = getattr(cls._model, "visual_projection", None)
        if projection is None:
            return pooled
        in_features = getattr(projection, "in_features", None)
        if in_features is not None and pooled.shape[-1] == in_features:
            return projection(pooled)
        return pooled

    @classmethod
    def _coerce_embedding_tensor(cls, raw_output, torch):
        if hasattr(raw_output, "image_embeds") and raw_output.image_embeds is not None:
            return raw_output.image_embeds
        if hasattr(raw_output, "pooler_output") and raw_output.pooler_output is not None:
            return cls._project_if_needed(raw_output.pooler_output, torch)
        if isinstance(raw_output, (tuple, list)) and raw_output:
            first_item = raw_output[0]
            if hasattr(first_item, "shape"):
                return first_item
        if hasattr(raw_output, "shape"):
            return raw_output
        raise RuntimeError(
            "Não foi possível converter a saída do modelo CLIP em embedding. "
            "Verifique a versão do transformers/torch instalada."
        )

    @classmethod
    def extract(cls, image_path: Path) -> list[float]:
        cls._bootstrap()
        torch, _, _ = DependencyManager.import_ai_stack()
        image = PillowImageLoader.load_rgb(image_path)
        inputs = cls._processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(cls._device)
        with torch.no_grad():
            try:
                raw_output = cls._model.get_image_features(pixel_values=pixel_values)
                embedding = cls._coerce_embedding_tensor(raw_output, torch)
            except Exception:
                vision_output = cls._model.vision_model(pixel_values=pixel_values)
                embedding = cls._coerce_embedding_tensor(vision_output, torch)
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
        return embedding[0].detach().cpu().tolist()


class FeatureExtractorFactory:
    @staticmethod
    def extract(image_path: Path, backend: str) -> list[float]:
        if backend == AI_BACKEND:
            return AIEmbeddingExtractor.extract(image_path)
        return ClassicFeatureExtractor.extract(image_path)


class ImageRepository:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def add_sample(self, label: str, source_path: Path, features: list[float], backend: str) -> Path:
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
                "INSERT INTO samples (label, image_path, features, backend) VALUES (?, ?, ?, ?)",
                (label.strip(), str(target_path.relative_to(APP_DIR)), json.dumps(features), backend),
            )
            conn.commit()
        return target_path

    def list_samples(self, backend: str | None = None) -> list[Sample]:
        query = "SELECT label, image_path, features, backend FROM samples"
        params: tuple[object, ...] = ()
        if backend is not None:
            query += " WHERE backend = ?"
            params = (backend,)
        query += " ORDER BY label, created_at DESC"
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            Sample(label=row[0], image_path=APP_DIR / row[1], features=json.loads(row[2]), backend=row[3])
            for row in rows
        ]

    def summarize_counts(self) -> dict[str, int]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT backend, COUNT(*) FROM samples GROUP BY backend").fetchall()
        summary = {CLASSIC_BACKEND: 0, AI_BACKEND: 0}
        for backend, count in rows:
            summary[backend] = count
        return summary


class RecognitionService:
    def __init__(self, repository: ImageRepository) -> None:
        self.repository = repository

    def recognize(self, image_path: Path, backend: str) -> dict[str, object] | None:
        samples = self.repository.list_samples(backend=backend)
        if not samples:
            return None

        query_features = FeatureExtractorFactory.extract(image_path, backend)
        scored: list[tuple[float, Sample]] = []
        for sample in samples:
            distance = self._distance(query_features, sample.features, backend)
            scored.append((distance, sample))

        scored.sort(key=lambda item: item[0])
        best_distance, best_sample = scored[0]
        confidence = self._distance_to_confidence(best_distance, backend)
        alternatives = [
            {
                "label": sample.label,
                "distance": round(distance, 4),
                "confidence": round(self._distance_to_confidence(distance, backend), 2),
                "backend": sample.backend,
            }
            for distance, sample in scored[:5]
        ]
        return {
            "label": best_sample.label,
            "distance": round(best_distance, 4),
            "confidence": round(confidence, 2),
            "sample_path": best_sample.image_path,
            "backend": backend,
            "alternatives": alternatives,
        }

    @staticmethod
    def _distance(first: Iterable[float], second: Iterable[float], backend: str) -> float:
        if backend == AI_BACKEND:
            dot = sum(a * b for a, b in zip(first, second, strict=True))
            return 1.0 - dot
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second, strict=True)))

    @staticmethod
    def _distance_to_confidence(distance: float, backend: str) -> float:
        if backend == AI_BACKEND:
            return max(0.0, min(100.0, (1.0 - distance) * 100.0))
        return max(0.0, min(100.0, 100.0 * math.exp(-1.4 * distance)))


class ModernImageRecognitionApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        DependencyManager.ensure_pillow()
        ensure_storage()
        self.title("Imagem Inteligente • Cadastro e Reconhecimento")
        self.geometry("1240x760")
        self.minsize(1040, 680)
        self.configure(bg="#0f172a")

        self.repository = ImageRepository(DB_PATH)
        self.recognition_service = RecognitionService(self.repository)
        self.selected_train_images: list[Path] = []
        self.selected_recognition_image: Path | None = None
        self._preview_photo = None
        self.mode_var = tk.StringVar(value=AI_BACKEND)
        self.backend_info_var = tk.StringVar(value="")
        self.dataset_info_var = tk.StringVar(value="")
        self._last_backend_warning: str | None = None

        self._setup_style()
        self._build_layout()
        self.refresh_dataset_summary()
        self.refresh_backend_status(show_message=False)

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
        style.configure("TRadiobutton", background="#111827", foreground="#e5e7eb")

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
            text="Agora com modo IA via CLIP para embeddings semânticos e fallback clássico para ambientes sem a stack de IA.",
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

    def _build_mode_selector(self, parent: ttk.Frame) -> None:
        mode_frame = ttk.Frame(parent, style="Card.TFrame")
        mode_frame.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        ttk.Label(mode_frame, text="Motor de reconhecimento:", style="Section.TLabel").pack(anchor="w")
        ttk.Radiobutton(
            mode_frame,
            text="IA (CLIP embeddings)",
            value=AI_BACKEND,
            variable=self.mode_var,
            command=self.refresh_backend_status,
        ).pack(anchor="w", pady=(8, 0))
        ttk.Radiobutton(
            mode_frame,
            text="Clássico (hash + histograma + bordas)",
            value=CLASSIC_BACKEND,
            variable=self.mode_var,
            command=self.refresh_backend_status,
        ).pack(anchor="w", pady=(4, 0))
        ttk.Label(mode_frame, textvariable=self.backend_info_var).pack(anchor="w", pady=(8, 0))
        ttk.Label(mode_frame, textvariable=self.dataset_info_var).pack(anchor="w", pady=(4, 0))

    def _build_training_section(self, parent: ttk.Frame) -> None:
        self._build_mode_selector(parent)
        ttk.Label(parent, text="1. Cadastro para treinamento", style="Section.TLabel").grid(row=1, column=0, sticky="w")
        ttk.Label(
            parent,
            text="Cadastre imagens semelhantes para treinar a base do backend selecionado.",
        ).grid(row=2, column=0, sticky="w", pady=(6, 12))

        form = ttk.Frame(parent, style="Card.TFrame")
        form.grid(row=3, column=0, sticky="ew")
        form.columnconfigure(1, weight=1)

        ttk.Label(form, text="Nome da categoria:").grid(row=0, column=0, sticky="w")
        self.label_entry = ttk.Entry(form)
        self.label_entry.grid(row=0, column=1, sticky="ew", padx=(12, 0))

        actions = ttk.Frame(parent, style="Card.TFrame")
        actions.grid(row=4, column=0, sticky="ew", pady=(12, 12))
        ttk.Button(actions, text="Selecionar imagens", style="Accent.TButton", command=self.select_training_images).pack(side="left")
        ttk.Button(actions, text="Cadastrar no sistema", command=self.save_training_samples).pack(side="left", padx=(10, 0))
        ttk.Button(actions, text="Atualizar lista", command=self.refresh_dataset_summary).pack(side="left", padx=(10, 0))

        self.selected_files_label = ttk.Label(parent, text="Nenhuma imagem selecionada.")
        self.selected_files_label.grid(row=5, column=0, sticky="w", pady=(0, 12))

        columns = ("categoria", "arquivo", "motor")
        self.dataset_tree = ttk.Treeview(parent, columns=columns, show="headings", height=12)
        self.dataset_tree.heading("categoria", text="Categoria")
        self.dataset_tree.heading("arquivo", text="Arquivo")
        self.dataset_tree.heading("motor", text="Motor")
        self.dataset_tree.column("categoria", width=160, anchor="w")
        self.dataset_tree.column("arquivo", width=330, anchor="w")
        self.dataset_tree.column("motor", width=110, anchor="center")
        self.dataset_tree.grid(row=6, column=0, sticky="nsew")
        parent.rowconfigure(6, weight=1)

    def _build_recognition_section(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="2. Reconhecimento", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            parent,
            text="Escolha uma imagem e compare usando o motor ativo no momento.",
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
            height=14,
            bg="#0b1220",
            fg="#e5e7eb",
            insertbackground="#e5e7eb",
            relief="flat",
            font=("Consolas", 10),
            wrap="word",
        )
        self.result_text.grid(row=4, column=0, sticky="ew")
        self._set_result_text(
            "Sugestão de uso:\n"
            "• Para IA, cadastre amostras no modo CLIP e reconheça também nesse modo.\n"
            "• Para um resultado semântico melhor, use classes bem definidas e mais imagens por classe.\n"
            "• Se a stack de IA não estiver instalada, o app volta automaticamente ao modo clássico."
        )

    def _set_result_text(self, text: str) -> None:
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert("1.0", text)
        self.result_text.configure(state="disabled")

    def get_active_backend(self) -> str:
        status = BackendManager.get_status(self.mode_var.get())
        return status.active_backend

    def refresh_backend_status(self, show_message: bool = True) -> None:
        status = BackendManager.get_status(self.mode_var.get())
        self.backend_info_var.set(f"Status do motor: {status.description}")
        if show_message and not status.ready and self._last_backend_warning != status.description:
            messagebox.showwarning("Modo IA indisponível", status.description)
            self._last_backend_warning = status.description
        if status.ready:
            self._last_backend_warning = None
        self.refresh_dataset_summary()

    def refresh_dataset_summary(self) -> None:
        for item in self.dataset_tree.get_children():
            self.dataset_tree.delete(item)
        for sample in self.repository.list_samples():
            motor = "IA" if sample.backend == AI_BACKEND else "Clássico"
            self.dataset_tree.insert("", "end", values=(sample.label, sample.image_path.name, motor))
        counts = self.repository.summarize_counts()
        self.dataset_info_var.set(
            f"Base treinada: {counts.get(AI_BACKEND, 0)} amostras IA | {counts.get(CLASSIC_BACKEND, 0)} amostras clássicas"
        )

    def select_training_images(self) -> None:
        files = filedialog.askopenfilenames(
            title="Selecione as imagens de treinamento",
            filetypes=SUPPORTED_FILE_TYPES,
        )
        self.selected_train_images = [Path(path) for path in files if Path(path).suffix.lower() in IMAGE_EXTENSIONS]
        if self.selected_train_images:
            names = ", ".join(path.name for path in self.selected_train_images[:3])
            extra = "" if len(self.selected_train_images) <= 3 else f" ... (+{len(self.selected_train_images) - 3})"
            backend_label = "IA" if self.get_active_backend() == AI_BACKEND else "Clássico"
            self.selected_files_label.config(
                text=f"{len(self.selected_train_images)} imagem(ns) para o motor {backend_label}: {names}{extra}"
            )
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

        backend = self.get_active_backend()
        saved_count = 0
        ignored_count = 0
        for image_path in self.selected_train_images:
            try:
                features = FeatureExtractorFactory.extract(image_path, backend)
                self.repository.add_sample(label, image_path, features, backend)
                saved_count += 1
            except ValueError as exc:
                ignored_count += 1
                messagebox.showwarning("Imagem ignorada", f"{image_path.name}: {exc}")
            except RuntimeError as exc:
                messagebox.showerror("Dependência ausente", str(exc))
                return

        self.label_entry.delete(0, tk.END)
        self.selected_train_images = []
        self.selected_files_label.config(text="Nenhuma imagem selecionada.")
        self.refresh_dataset_summary()
        backend_name = "IA (CLIP)" if backend == AI_BACKEND else "Clássico"
        messagebox.showinfo(
            "Treinamento atualizado",
            f"{saved_count} imagem(ns) cadastrada(s) na categoria '{label}' usando o motor {backend_name}. "
            f"Ignoradas: {ignored_count}.",
        )

    def select_recognition_image(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Escolha uma imagem para reconhecimento",
            filetypes=SUPPORTED_FILE_TYPES,
        )
        if not file_path:
            return
        self.selected_recognition_image = Path(file_path)
        self._load_preview(self.selected_recognition_image)

    def _load_preview(self, image_path: Path) -> None:
        try:
            self._preview_photo = PillowImageLoader.load_preview(image_path)
        except ValueError as exc:
            messagebox.showwarning("Formato não suportado", str(exc))
            return
        self.preview_label.configure(image=self._preview_photo, text="")

    def run_recognition(self) -> None:
        if not self.selected_recognition_image:
            messagebox.showwarning("Imagem obrigatória", "Escolha uma imagem para reconhecimento.")
            return

        backend = self.get_active_backend()
        try:
            result = self.recognition_service.recognize(self.selected_recognition_image, backend)
        except ValueError as exc:
            messagebox.showwarning("Erro no reconhecimento", str(exc))
            return
        except RuntimeError as exc:
            messagebox.showerror("Dependência ausente", str(exc))
            return

        if result is None:
            backend_name = "IA (CLIP)" if backend == AI_BACKEND else "Clássico"
            messagebox.showwarning(
                "Base vazia",
                f"Cadastre imagens de treinamento para o motor {backend_name} antes de reconhecer.",
            )
            return

        confidence = result["confidence"]
        decision = "Correspondência forte" if confidence >= 70 else "Correspondência aproximada"
        backend_name = "IA (CLIP)" if backend == AI_BACKEND else "Clássico"
        alternative_lines = [
            f"- {item['label']}: confiança {item['confidence']}% | distância {item['distance']}"
            for item in result["alternatives"]
        ]
        output = (
            f"Motor ativo: {backend_name}\n"
            f"Resultado principal: {result['label']}\n"
            f"Nível de confiança: {confidence}%\n"
            f"Distância vetorial: {result['distance']}\n"
            f"Status: {decision}\n\n"
            "Top correspondências:\n"
            + "\n".join(alternative_lines)
        )
        self._set_result_text(output)


def main() -> None:
    try:
        app = ModernImageRecognitionApp()
    except RuntimeError as exc:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Dependência ausente", str(exc))
        root.destroy()
        raise SystemExit(1) from exc
    app.mainloop()


if __name__ == "__main__":
    main()
