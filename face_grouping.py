"""Group event images that contain the face from a captured or uploaded image."""

from __future__ import annotations

import re
import shutil
import threading
import warnings
from pathlib import Path
from tkinter import Tk, filedialog, messagebox, StringVar
from tkinter.ttk import Button, Entry, Label, Frame
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import onnxruntime

onnxruntime.set_default_logger_severity(3)
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")
warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime")


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PROJECT_DIR = Path(__file__).resolve().parent
EVENT_IMAGES_DIR = PROJECT_DIR / "media" / "Event Images"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "media" / "grouped"


class FaceGrouper:
    def __init__(self, similarity_threshold: float = 0.45) -> None:
        self.similarity_threshold = similarity_threshold
        self.face_app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )
        self.face_app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.35)

    def _embeddings(self, image_path: Path) -> list[np.ndarray]:
        image = cv2.imread(str(image_path))
        if image is None:
            return []
        faces = self.face_app.get(image)
        return [face.embedding for face in faces]

    @staticmethod
    def _normalise(embedding: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm

    def group_images(self, reference_image: Path, event_images_dir: Path, output_dir: Path) -> tuple[int, int]:
        reference_embeddings = [
            self._normalise(embedding)
            for embedding in self._embeddings(reference_image)
        ]
        if not reference_embeddings:
            raise ValueError("No face was found in the selected reference image.")

        event_files = sorted(
            path for path in event_images_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            and "guests_profiles" not in path.parts
        )
        if not event_files:
            raise ValueError(f"No images were found in: {event_images_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)
        matched = 0
        readable = 0

        for event_file in event_files:
            event_embeddings = self._embeddings(event_file)
            if not event_embeddings:
                continue
            readable += 1

            best_score = max(
                float(np.dot(self._normalise(event_embedding), reference_embedding))
                for event_embedding in event_embeddings
                for reference_embedding in reference_embeddings
            )
            if best_score < self.similarity_threshold:
                continue

            destination = output_dir / event_file.name
            if destination.exists():
                destination = output_dir / f"{event_file.stem}_{matched + 1}{event_file.suffix}"
            shutil.copy2(event_file, destination)
            matched += 1

        return matched, readable


class FaceGroupingApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("Face Grouping")
        self.root.resizable(False, False)
        self.reference_path = StringVar()
        self.output_name = StringVar()
        self.status = StringVar(value=f"Event images: {EVENT_IMAGES_DIR}")

        frame = Frame(root, padding=16)
        frame.grid(sticky="nsew")

        Label(frame, text="Reference face").grid(row=0, column=0, sticky="w", pady=4)
        Entry(frame, textvariable=self.reference_path, width=54).grid(row=1, column=0, columnspan=2, pady=4)
        Button(frame, text="Take Picture", command=self.take_picture).grid(row=2, column=0, sticky="w", pady=4)
        Button(frame, text="Upload Picture", command=self.upload_picture).grid(row=2, column=1, sticky="e", pady=4)

        Label(frame, text="Folder name").grid(row=3, column=0, sticky="w", pady=(12, 4))
        Entry(frame, textvariable=self.output_name, width=54).grid(row=4, column=0, columnspan=2, pady=4)
        Button(frame, text="Analyze Event Images", command=self.start_grouping).grid(row=5, column=0, columnspan=2, pady=12)
        Label(frame, textvariable=self.status, wraplength=440).grid(row=6, column=0, columnspan=2, sticky="w")

    def upload_picture(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select a reference face image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")],
        )
        if selected:
            self.reference_path.set(selected)

    def take_picture(self) -> None:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            messagebox.showerror("Camera error", "Could not open the camera.")
            return

        captured = None
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            cv2.imshow("Take Picture - press SPACE to capture or ESC to cancel", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 32:
                captured = frame
                break
            if key == 27:
                break

        camera.release()
        cv2.destroyAllWindows()
        if captured is not None:
            capture_path = Path(__file__).resolve().parent / "face_grouping_reference.jpg"
            cv2.imwrite(str(capture_path), captured)
            self.reference_path.set(str(capture_path))

    def start_grouping(self) -> None:
        reference = Path(self.reference_path.get().strip())
        folder_name = self.output_name.get().strip()
        if not reference.is_file():
            messagebox.showerror("Missing reference", "Take or upload a face picture first.")
            return
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _-]*", folder_name):
            messagebox.showerror("Invalid folder name", "Use letters, numbers, spaces, hyphens, or underscores.")
            return

        output_dir = DEFAULT_OUTPUT_DIR / folder_name
        self.status.set("Loading face model and analyzing images...")
        threading.Thread(
            target=self._run_grouping,
            args=(reference, output_dir),
            daemon=True,
        ).start()

    def _run_grouping(self, reference: Path, output_dir: Path) -> None:
        try:
            grouper = FaceGrouper()
            matched, readable = grouper.group_images(reference, EVENT_IMAGES_DIR, output_dir)
            self.root.after(0, lambda: self.status.set(f"Copied {matched} matching images to {output_dir} ({readable} readable event images)."))
        except Exception as error:
            self.root.after(0, lambda: messagebox.showerror("Grouping failed", str(error)))
            self.root.after(0, lambda: self.status.set("Ready"))


if __name__ == "__main__":
    application = Tk()
    FaceGroupingApp(application)
    application.mainloop()