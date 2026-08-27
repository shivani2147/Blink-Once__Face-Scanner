from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pillow_heif import register_heif_opener

from face_grouping import FaceGrouper, IMAGE_EXTENSIONS


PROJECT_DIR = Path(__file__).resolve().parent
EVENT_IMAGES_DIR = PROJECT_DIR / "Event Images_"
MEDIA_DIR = PROJECT_DIR / "media"
GROUPED_DIR = MEDIA_DIR / "grouped"
EVENTS_DIR = MEDIA_DIR / "events"
HEIC_EXTENSIONS = {".heic", ".heif"}
register_heif_opener()
EVENTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Gathered Face Grouping API")
app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")
app.mount("/event-images", StaticFiles(directory=EVENT_IMAGES_DIR), name="event-images")
app.mount("/uploaded-events", StaticFiles(directory=EVENTS_DIR), name="uploaded-events")


@app.get("/", include_in_schema=False)
def homepage() -> FileResponse:
    return FileResponse(PROJECT_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def event_slug(event_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", event_name).strip("-").lower()
    if not slug:
        raise HTTPException(status_code=400, detail="Enter a valid event name.")
    return slug


@app.post("/api/events")
def create_event(
    event_name: str = Form(...),
    event_date: str = Form(""),
    photos: list[UploadFile] = File(...),
) -> dict[str, object]:
    slug = event_slug(event_name)
    event_dir = EVENTS_DIR / slug
    event_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    skipped: list[dict[str, str]] = []

    for photo in photos:
        if not photo.filename:
            skipped.append({"name": "unnamed file", "reason": "Missing filename"})
            continue
        filename = Path(photo.filename).name
        extension = Path(filename).suffix.lower()
        if extension not in IMAGE_EXTENSIONS and extension not in HEIC_EXTENSIONS:
            skipped.append({"name": filename, "reason": "Unsupported image format"})
            continue
        if extension in HEIC_EXTENSIONS:
            filename = f"{Path(filename).stem}.jpg"
        destination = event_dir / filename
        duplicate_number = 1
        while destination.exists():
            destination = event_dir / f"{Path(filename).stem}_{duplicate_number}{Path(filename).suffix}"
            duplicate_number += 1
        if extension in HEIC_EXTENSIONS:
            with Image.open(photo.file) as image:
                image.convert("RGB").save(destination, format="JPEG", quality=95)
        else:
            with destination.open("wb") as output:
                shutil.copyfileobj(photo.file, output)
        saved += 1

    if saved == 0:
        raise HTTPException(status_code=400, detail="Upload at least one image for this event.")

    return {
        "name": event_name,
        "date": event_date,
        "slug": slug,
        "photos": saved,
        "skipped": skipped,
        "url": f"/?event={quote(slug)}",
    }


@app.post("/api/group")
def group_photos(
    reference: UploadFile = File(...),
    folder_name: str = Form(...),
    event: str = Form("All events"),
) -> dict[str, object]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _-]*", folder_name):
        raise HTTPException(status_code=400, detail="Use a valid folder name.")

    if not reference.content_type or not reference.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="The reference file must be an image.")

    normalized_event = event_slug(event) if event not in {"", "All events"} else event
    event_dir = EVENTS_DIR / normalized_event if normalized_event not in {"", "All events"} else EVENT_IMAGES_DIR
    if not event_dir.is_dir():
        event_dir = EVENT_IMAGES_DIR / event if event != "All events" else EVENT_IMAGES_DIR
    if not event_dir.is_dir():
        event_dir = EVENT_IMAGES_DIR

    output_dir = GROUPED_DIR / folder_name
    suffix = Path(reference.filename or "reference.jpg").suffix.lower() or ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=PROJECT_DIR) as temporary_file:
        temporary_path = Path(temporary_file.name)

    try:
        with temporary_path.open("wb") as destination:
            shutil.copyfileobj(reference.file, destination)

        grouper = FaceGrouper()
        matched, readable = grouper.group_images(temporary_path, event_dir, output_dir)
        image_urls = [
            f"/media/grouped/{folder_name}/{image_file.name}"
            for image_file in sorted(output_dir.iterdir())
            if image_file.is_file() and image_file.suffix.lower() in IMAGE_EXTENSIONS
        ]
        return {"matched": matched, "readable": readable, "images": image_urls}
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Face grouping failed: {error}") from error
    finally:
        temporary_path.unlink(missing_ok=True)
