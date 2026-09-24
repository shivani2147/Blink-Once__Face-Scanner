from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import quote, unquote

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Body, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pillow_heif import register_heif_opener

from face_grouping import FaceGrouper, IMAGE_EXTENSIONS


PROJECT_DIR = Path(__file__).resolve().parent
EVENT_IMAGES_DIR = PROJECT_DIR / "Event Images_"
MEDIA_DIR = PROJECT_DIR / "media"
GROUPED_DIR = MEDIA_DIR / "grouped"
EVENTS_DIR = MEDIA_DIR / "events"
THUMBNAIL_CACHE_DIR = MEDIA_DIR / "cache"
LIVE_STATE_FILE = MEDIA_DIR / "live_state.json"
HEIC_EXTENSIONS = {".heic", ".heif"}
register_heif_opener()
EVENT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
EVENTS_DIR.mkdir(parents=True, exist_ok=True)
THUMBNAIL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Gathered Face Grouping API")
app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")
app.mount("/event-images", StaticFiles(directory=EVENT_IMAGES_DIR), name="event-images")
app.mount("/uploaded-events", StaticFiles(directory=EVENTS_DIR), name="uploaded-events")


@app.get("/gallery")
def get_gallery_page() -> FileResponse:
    return FileResponse(PROJECT_DIR / "gallery.html")


@app.get("/api/thumbnail")
def get_thumbnail(src: str, w: int = 400, q: int = 80) -> FileResponse:
    if not src:
        raise HTTPException(status_code=400, detail="Missing src parameter")

    clean_src = src.lstrip("/")
    if clean_src.startswith("uploaded-events/"):
        rel_path = clean_src.replace("uploaded-events/", "", 1)
        file_path = (EVENTS_DIR / rel_path).resolve()
        if EVENTS_DIR.resolve() not in file_path.parents and file_path != EVENTS_DIR.resolve():
            raise HTTPException(status_code=400, detail="Invalid path")
    elif clean_src.startswith("event-images/"):
        rel_path = clean_src.replace("event-images/", "", 1)
        file_path = (EVENT_IMAGES_DIR / rel_path).resolve()
        if EVENT_IMAGES_DIR.resolve() not in file_path.parents:
            raise HTTPException(status_code=400, detail="Invalid path")
    elif clean_src.startswith("media/"):
        rel_path = clean_src.replace("media/", "", 1)
        file_path = (MEDIA_DIR / rel_path).resolve()
        if MEDIA_DIR.resolve() not in file_path.parents:
            raise HTTPException(status_code=400, detail="Invalid path")
    else:
        raise HTTPException(status_code=400, detail="Invalid source directory")

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    cache_key = hashlib.md5(f"{file_path}_{w}_{q}".encode("utf-8")).hexdigest()
    cache_file = THUMBNAIL_CACHE_DIR / f"{cache_key}.jpg"

    if cache_file.exists():
        if cache_file.stat().st_mtime >= file_path.stat().st_mtime:
            return FileResponse(cache_file, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})

    try:
        with Image.open(file_path) as img:
            img = img.convert("RGB")
            img.thumbnail((w, w), Image.Resampling.LANCZOS)
            img.save(cache_file, format="JPEG", quality=q, optimize=True)
        return FileResponse(cache_file, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})
    except Exception:
        return FileResponse(file_path)


@app.get("/", include_in_schema=False)
def homepage() -> FileResponse:
    return FileResponse(PROJECT_DIR / "index.html")

@app.get("/admin", include_in_schema=False)
def admin_page() -> FileResponse:
    return FileResponse(PROJECT_DIR / "admin.html")

@app.get("/admin/event", include_in_schema=False)
def event_admin_page() -> FileResponse:
    return FileResponse(PROJECT_DIR / "event_admin.html")

@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def event_slug(event_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", event_name).strip("-").lower()
    if not slug:
        raise HTTPException(status_code=400, detail="Enter a valid event name.")
    return slug


@app.get("/api/admin/events")
def get_events() -> dict[str, object]:
    events = []
    if not EVENTS_DIR.exists():
        return {"events": events}
        
    for event_dir in EVENTS_DIR.iterdir():
        if event_dir.is_dir():
            metadata_file = event_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    try:
                        metadata = json.load(f)
                    except json.JSONDecodeError:
                        continue
                
                # Count photos
                photos_count = sum(
                    1 for f in event_dir.iterdir()
                    if f.is_file() and (f.suffix.lower() in IMAGE_EXTENSIONS or f.suffix.lower() in HEIC_EXTENSIONS)
                )
                
                # Fix cover photo path with optimized thumbnail
                cover_photo = metadata.get("cover_photo", "")
                if cover_photo:
                    raw_cover = f"/uploaded-events/{event_dir.name}/{cover_photo}"
                    cover_photo = f"/api/thumbnail?src={quote(raw_cover)}&w=500"
                
                events.append({
                    "slug": event_dir.name,
                    "type": metadata.get("type", "EVENT"),
                    "client_name": metadata.get("client_name", event_dir.name),
                    "date_range": metadata.get("date_range", ""),
                    "venue": metadata.get("venue", ""),
                    "cover_photo": cover_photo,
                    "photos_count": photos_count,
                    "guest_count": len(metadata.get("guests", []))
                })
    
    events.sort(key=lambda x: x["client_name"])
    return {"events": events}


@app.get("/api/admin/gallery")
def get_gallery(skip: int = 0, limit: int = 100) -> dict[str, object]:
    if not EVENT_IMAGES_DIR.exists():
        return {"images": []}
    
    # Sort files by modification time descending (newest first)
    files = [f for f in EVENT_IMAGES_DIR.iterdir() if f.is_file() and (f.suffix.lower() in IMAGE_EXTENSIONS or f.suffix.lower() in HEIC_EXTENSIONS)]
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Paginate
    paginated_files = files[skip : skip + limit]
    image_urls = [f"/api/thumbnail?src={quote(f'/event-images/{f.name}')}&w=400" for f in paginated_files]
    
    return {
        "images": image_urls,
        "total": len(files)
    }


@app.delete("/api/admin/photos")
def clear_photos() -> dict[str, object]:
    if not EVENT_IMAGES_DIR.exists():
        return {"status": "success", "deleted": 0}
    deleted = 0
    for file in EVENT_IMAGES_DIR.iterdir():
        if file.is_file():
            try:
                file.unlink()
                deleted += 1
            except Exception:
                pass
    return {"status": "success", "deleted": deleted}


@app.get("/api/admin/events/{slug}/photos")
def get_event_photos(slug: str) -> dict[str, object]:
    event_dir = EVENTS_DIR / slug
    if not event_dir.exists() or not event_dir.is_dir():
        raise HTTPException(status_code=404, detail="Event not found")
        
    files = [f for f in event_dir.iterdir() if f.is_file() and (f.suffix.lower() in IMAGE_EXTENSIONS or f.suffix.lower() in HEIC_EXTENSIONS)]
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    images = []
    for f in files:
        raw_url = f"/uploaded-events/{slug}/{f.name}"
        images.append({
            "filename": f.name,
            "url": raw_url,
            "thumbnail": f"/api/thumbnail?src={quote(raw_url)}&w=400"
        })
        
    return {"images": images}


@app.delete("/api/admin/events/{slug}/photos")
def delete_event_photos(slug: str, filenames: list[str] = Body(...)) -> dict[str, object]:
    event_dir = EVENTS_DIR / slug
    if not event_dir.exists() or not event_dir.is_dir():
        raise HTTPException(status_code=404, detail="Event not found")
        
    deleted = 0
    errors = []
    for filename in filenames:
        # Prevent directory traversal
        file_path = (event_dir / filename).resolve()
        if event_dir.resolve() not in file_path.parents:
            errors.append(f"Invalid filename: {filename}")
            continue
            
        if file_path.exists() and file_path.is_file():
            try:
                file_path.unlink()
                deleted += 1
            except Exception as e:
                errors.append(f"Failed to delete {filename}: {str(e)}")
        else:
            errors.append(f"File not found: {filename}")
            
    return {"deleted": deleted, "errors": errors}


@app.delete("/api/admin/events/{slug}")
def delete_event(slug: str) -> dict[str, object]:
    event_dir = EVENTS_DIR / slug
    if not event_dir.exists() or not event_dir.is_dir():
        raise HTTPException(status_code=404, detail="Event not found")
        
    try:
        shutil.rmtree(event_dir)
        return {"status": "success", "deleted": slug}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete event: {str(e)}")



@app.get("/api/admin/live-event")
def get_live_event() -> dict[str, str | None]:
    if LIVE_STATE_FILE.exists():
        try:
            with open(LIVE_STATE_FILE, "r") as f:
                data = json.load(f)
                return {"live_slug": data.get("live_slug")}
        except Exception:
            pass
    return {"live_slug": None}


@app.get("/api/admin/events/{slug}/guests")
def get_event_guests(slug: str) -> dict[str, object]:
    event_dir = EVENTS_DIR / slug
    if not event_dir.exists() or not event_dir.is_dir():
        raise HTTPException(status_code=404, detail="Event not found")
        
    metadata_file = event_dir / "metadata.json"
    guests = []
    if metadata_file.exists():
        try:
            with open(metadata_file, "r") as f:
                meta = json.load(f)
            raw_guests = meta.get("guests", [])
            for g in raw_guests:
                if isinstance(g, dict):
                    guests.append(g)
                else:
                    guests.append({"name": g, "contact": "", "upcoming_event": "", "image": ""})
        except Exception:
            pass
            
    return {"guests": guests}


@app.post("/api/admin/live-event")
def set_live_event(payload: dict[str, str | None] = Body(...)) -> dict[str, str | None]:
    slug = payload.get("live_slug")
    if slug is not None:
        event_dir = EVENTS_DIR / slug
        if not event_dir.exists() or not event_dir.is_dir():
            raise HTTPException(status_code=404, detail="Event not found")
            
    with open(LIVE_STATE_FILE, "w") as f:
        json.dump({"live_slug": slug}, f)
        
    return {"live_slug": slug}


@app.post("/api/admin/events/{slug}/photos")
def upload_event_photos(
    slug: str,
    photos: list[UploadFile] = File(...),
) -> dict[str, object]:
    event_dir = EVENTS_DIR / slug
    if not event_dir.exists() or not event_dir.is_dir():
        raise HTTPException(status_code=404, detail="Event not found")

    saved = 0
    skipped: list[dict[str, str]] = []
    first_saved_file = ""

    for photo in photos:
        if not photo.filename:
            skipped.append({
                "name": "unnamed file",
                "reason": "Missing filename"
            })
            continue

        filename = Path(photo.filename).name
        extension = Path(filename).suffix.lower()

        if extension not in IMAGE_EXTENSIONS and extension not in HEIC_EXTENSIONS:
            skipped.append({
                "name": filename,
                "reason": "Unsupported image format"
            })
            continue

        if extension in HEIC_EXTENSIONS:
            filename = f"{Path(filename).stem}.jpg"

        destination = event_dir / filename

        duplicate_number = 1
        while destination.exists():
            destination = event_dir / (
                f"{Path(filename).stem}_{duplicate_number}"
                f"{Path(filename).suffix}"
            )
            duplicate_number += 1

        final_filename = destination.name

        try:
            if extension in HEIC_EXTENSIONS:
                with Image.open(photo.file) as image:
                    image.convert("RGB").save(
                        destination,
                        format="JPEG",
                        quality=95
                    )
            else:
                with destination.open("wb") as output:
                    shutil.copyfileobj(photo.file, output)

        except Image.DecompressionBombError:
            skipped.append({
                "name": photo.filename,
                "reason": "Image resolution is too large to process safely"
            })
            continue

        except Exception as e:
            skipped.append({
                "name": photo.filename,
                "reason": f"Failed to process image: {str(e)}"
            })
            continue

        if not first_saved_file:
            first_saved_file = final_filename

        saved += 1

    if saved == 0:
        raise HTTPException(
            status_code=400,
            detail="Upload at least one valid image or media file."
        )

    metadata_file = event_dir / "metadata.json"

    if metadata_file.exists():
        try:
            with open(metadata_file, "r") as f:
                meta = json.load(f)

            if not meta.get("cover_photo") and first_saved_file:
                meta["cover_photo"] = first_saved_file

                with open(metadata_file, "w") as f:
                    json.dump(meta, f)

        except Exception:
            pass

    return {
        "slug": slug,
        "saved": saved,
        "skipped": skipped,
    }


@app.get("/api/admin/events/{slug}/download")
@app.get("/api/events/{slug}/download")
def download_event_photos(slug: str, background_tasks: BackgroundTasks) -> FileResponse:
    event_dir = EVENTS_DIR / slug
    if not event_dir.exists() or not event_dir.is_dir():
        raise HTTPException(status_code=404, detail="Event not found")

    files = [
        f for f in event_dir.iterdir()
        if f.is_file() and (f.suffix.lower() in IMAGE_EXTENSIONS or f.suffix.lower() in HEIC_EXTENSIONS)
    ]
    if not files:
        raise HTTPException(status_code=400, detail="No photos available to download")

    fd, temp_path = tempfile.mkstemp(suffix=".zip")
    os.close(fd)

    def cleanup():
        try:
            os.unlink(temp_path)
        except Exception:
            pass

    background_tasks.add_task(cleanup)

    with zipfile.ZipFile(temp_path, mode="w", compression=zipfile.ZIP_STORED) as zf:
        for file in files:
            arcname = file.stem + ".jpg"
            zf.write(file, arcname=arcname)

    headers = {
        "Access-Control-Expose-Headers": "Content-Length"
    }
    return FileResponse(temp_path, filename=f"{slug}-gallery.zip", media_type="application/zip", headers=headers)


@app.post("/api/group/download")
def download_grouped_photos(background_tasks: BackgroundTasks, payload: dict[str, list[str]] = Body(...)) -> FileResponse:
    image_urls = payload.get("images", [])
    if not image_urls:
        raise HTTPException(status_code=400, detail="No images provided for download")

    fd, temp_path = tempfile.mkstemp(suffix=".zip")
    os.close(fd)

    def cleanup():
        try:
            os.unlink(temp_path)
        except Exception:
            pass

    background_tasks.add_task(cleanup)

    with zipfile.ZipFile(temp_path, mode="w", compression=zipfile.ZIP_STORED) as zf:
        added_files = set()
        for url in image_urls:
            url = unquote(url)
            file_path = None
            if url.startswith("/media/grouped/"):
                relative_path = url.replace("/media/grouped/", "")
                file_path = (GROUPED_DIR / relative_path).resolve()
            elif url.startswith("/uploaded-events/"):
                relative_path = url.replace("/uploaded-events/", "")
                file_path = (EVENTS_DIR / relative_path).resolve()
                
            if file_path and file_path.exists() and file_path.is_file():
                arcname = file_path.stem + ".jpg"
                counter = 1
                while arcname in added_files:
                    arcname = f"{file_path.stem}_{counter}.jpg"
                    counter += 1
                
                zf.write(file_path, arcname=arcname)
                added_files.add(arcname)

    headers = {
        "Access-Control-Expose-Headers": "Content-Length"
    }
    return FileResponse(temp_path, filename="downloaded-photos.zip", media_type="application/zip", headers=headers)


@app.post("/api/events")
def upload_photos(
    event_type: str = Form("EVENT"),
    client_name: str = Form(...),
    date_range: str = Form(""),
    venue: str = Form(""),
    photos: list[UploadFile] = File(...),
) -> dict[str, object]:
    slug = event_slug(client_name)
    event_dir = EVENTS_DIR / slug
    event_dir.mkdir(parents=True, exist_ok=True)
    
    saved = 0
    skipped: list[dict[str, str]] = []
    cover_photo_name = ""

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
            
        final_filename = destination.name
            
        if extension in HEIC_EXTENSIONS:
            with Image.open(photo.file) as image:
                image.convert("RGB").save(destination, format="JPEG", quality=95)
        else:
            with destination.open("wb") as output:
                shutil.copyfileobj(photo.file, output)
        
        if not cover_photo_name:
            cover_photo_name = final_filename
            
        saved += 1

    if saved == 0:
        raise HTTPException(status_code=400, detail="Upload at least one image.")

    metadata = {
        "type": event_type,
        "client_name": client_name,
        "date_range": date_range,
        "venue": venue,
        "cover_photo": cover_photo_name
    }
    
    with open(event_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    return {
        "slug": slug,
        "photos": saved,
        "skipped": skipped,
    }


# Store job status for background tasks
JOB_STATUS: dict[str, dict[str, str]] = {}

def run_face_grouping(temporary_path: Path, event_dir: Path, output_dir: Path, job_key: str):
    try:
        grouper = FaceGrouper()
        grouper.group_images(temporary_path, event_dir, output_dir)
        JOB_STATUS[job_key] = {"status": "done", "error": ""}
    except Exception as error:
        JOB_STATUS[job_key] = {"status": "error", "error": str(error)}
    finally:
        temporary_path.unlink(missing_ok=True)

@app.post("/api/group")
def group_photos(
    background_tasks: BackgroundTasks,
    reference: UploadFile = File(...),
    folder_name: str = Form(...),
    event: str = Form(...),
    contact: str = Form(default=""),
    upcoming_event: str = Form(default=""),
) -> dict[str, object]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _-]*", folder_name):
        raise HTTPException(status_code=400, detail="Use a valid folder name.")

    if not reference.content_type or not reference.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="The reference file must be an image.")

    # Strictly enforce Live event rule
    live_event = get_live_event()["live_slug"]
    if live_event and event != live_event:
        event = live_event # Force live event
        
    event_dir = EVENTS_DIR / event_slug(event)
    if not event_dir.is_dir():
        raise HTTPException(status_code=404, detail="Event not found.")

    output_dir = GROUPED_DIR / event_slug(event) / folder_name
    # Clear output_dir if starting a new job to prevent mixing old photos
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(reference.filename or "reference.jpg").suffix.lower() or ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=PROJECT_DIR) as temporary_file:
        temporary_path = Path(temporary_file.name)

    try:
        with temporary_path.open("wb") as destination:
            shutil.copyfileobj(reference.file, destination)
            
        metadata_file = event_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    meta = json.load(f)
                
                guests = meta.get("guests", [])
                
                exists = False
                for g in guests:
                    if isinstance(g, dict) and g.get("name") == folder_name:
                        exists = True
                        break
                    elif isinstance(g, str) and g == folder_name:
                        exists = True
                        break
                        
                if not exists:
                    # Save a copy of the reference image for the profile picture
                    profile_pic_dir = event_dir / "guests_profiles"
                    profile_pic_dir.mkdir(parents=True, exist_ok=True)
                    profile_pic_path = profile_pic_dir / f"{folder_name}.jpg"
                    shutil.copy2(temporary_path, profile_pic_path)
                    
                    guests.append({
                        "name": folder_name,
                        "contact": contact,
                        "upcoming_event": upcoming_event,
                        "image": f"/api/thumbnail?src={quote(f'/uploaded-events/{event_slug(event)}/guests_profiles/{folder_name}.jpg')}&w=150"
                    })
                    meta["guests"] = guests
                    with open(metadata_file, "w") as f:
                        json.dump(meta, f)
            except Exception:
                pass

        job_key = f"{event_slug(event)}_{folder_name}"
        JOB_STATUS[job_key] = {"status": "processing", "error": ""}
        background_tasks.add_task(run_face_grouping, temporary_path, event_dir, output_dir, job_key)
        
        return {"status": "processing", "folder": folder_name, "event": event_slug(event)}
    except Exception as error:
        temporary_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to start job: {error}") from error

@app.get("/api/grouped/{event}/{folder_name}")
def get_grouped_photos(event: str, folder_name: str) -> dict[str, object]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _-]*", folder_name):
        raise HTTPException(status_code=400, detail="Invalid folder name.")
        
    output_dir = GROUPED_DIR / event / folder_name
    image_urls = []
    if output_dir.exists():
        image_urls = [
            f"/media/grouped/{event}/{folder_name}/{image_file.name}"
            for image_file in sorted(output_dir.iterdir(), key=lambda x: x.stat().st_mtime)
            if image_file.is_file() and image_file.suffix.lower() in IMAGE_EXTENSIONS
        ]
        
    job_key = f"{event}_{folder_name}"
    status_info = JOB_STATUS.get(job_key, {"status": "unknown", "error": ""})
    return {
        "status": status_info["status"],
        "error": status_info.get("error", ""),
        "images": image_urls
    }
