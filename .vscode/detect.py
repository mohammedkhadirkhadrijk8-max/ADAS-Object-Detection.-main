import shutil
import ssl
from pathlib import Path
from urllib import request

try:
    from ultralytics import YOLO
except ModuleNotFoundError as exc:
    raise SystemExit("Install the dependency first: pip install ultralytics") from exc

BASE_DIR = Path(__file__).resolve().parent.parent
VIDEO_PATH = BASE_DIR / "dashcam.mp4"
VIDEO_URL = "https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4"
OUTPUT_DIR = BASE_DIR / "output"


def download_video(video_url: str, destination: Path) -> None:
    if destination.exists():
        print(f"Video already exists at {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading sample dashcam video to {destination}...")

    req = request.Request(video_url, headers={"User-Agent": "Mozilla/5.0"})
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    with request.urlopen(req, timeout=120, context=ssl_context) as response, destination.open("wb") as video_file:
        shutil.copyfileobj(response, video_file)

    print("✅ Download complete!")


def main() -> None:
    download_video(VIDEO_URL, VIDEO_PATH)

    print("Loading YOLO model...")
    model = YOLO("yolov8n.pt")

    print("Processing video frames... This might take a minute.")
    model(
        source=str(VIDEO_PATH),
        save=True,
        project=str(OUTPUT_DIR),
        name="run1",
    )

    print(f"✅ Processing complete! Check the '{OUTPUT_DIR / 'run1'}' folder in VS Code.")


if __name__ == "__main__":
    main()
