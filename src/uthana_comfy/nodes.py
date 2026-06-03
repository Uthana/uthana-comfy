import json
import math
import os
import asyncio
import threading
import time
import io
import tempfile
import random
from pathlib import Path

from comfy.utils import ProgressBar
import folder_paths

CONFIG_PATH = Path(__file__).parent / "config.json"


def get_api_key():
    key = os.environ.get("UTHANA_API_KEY")
    if key:
        return key
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
            return cfg.get("api_key")
    return None


OUTPUT_FORMATS = ["GLB", "FBX"]


_CHARACTER_NAMES = ["tar", "ava", "manny", "quinn", "y_bot"]


def run_with_progress(fn, estimated_duration):
    result = [None]
    error = [None]

    def run():
        try:
            result[0] = fn()
        except Exception as e:
            error[0] = e

    thread = threading.Thread(target=run)
    thread.start()

    pbar = ProgressBar(100)
    start = time.monotonic()
    while thread.is_alive():
        thread.join(timeout=0.5)
        elapsed = time.monotonic() - start
        t = elapsed / estimated_duration
        if t < 0.9:
            progress = t / 0.9 * 90
        else:
            overshoot = elapsed - estimated_duration * 0.9
            progress = 100 - 10 * math.exp(-overshoot / 10)
        pbar.update_absolute(int(progress))

    pbar.update_absolute(100)

    if error[0] is not None:
        raise error[0]

    return result[0]


def resolve_character_id(value):
    try:
        import uthana
    except ImportError:
        raise ImportError("This node requires uthana. Install with: pip install uthana")

    resolved = getattr(uthana.UthanaCharacters(), value, None)
    if resolved is not None:
        return resolved
    if value.startswith("c"):
        return value
    raise uthana.Error(
        f"Invalid character_id: '{value}'. Use a name ({', '.join(_CHARACTER_NAMES)}) or a character ID starting with 'c'."
    )


class TextToMotion:
    """Generate motion from text prompt using Uthana text-to-motion vqvae-v1 model"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A person"}),
                "model": (["vqvae-v1", "diffusion-v2"], {"default": "vqvae-v1"}),
                "character_id": ("STRING", {"default": "tar"}),
                "foot_ik": ("BOOLEAN", {"default": True}),
                "cfg_scale": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 1.0}),
                "seed": ("INT", {"default": random.randint(1, 99999), "min": 1, "max": 99999, "step": 1}),
                "internal_ik": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("character_id", "motion_id")
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "Uthana"

    async def execute(self, prompt, model, character_id, foot_ik, cfg_scale, seed, internal_ik):
        try:
            import uthana
        except ImportError:
            raise ImportError("This node requires uthana. Install with: pip install uthana")

        client = uthana.Client(get_api_key())
        result = await client.ttm.create(
            prompt=prompt,
            model=model,
            character_id=resolve_character_id(character_id),
            foot_ik=foot_ik,
            cfg_scale=cfg_scale,
            seed=seed,
            internal_ik=internal_ik,
        )

        return (result.character_id, result.motion_id)


class VideoToMotion:
    """Generate motion from a video using Uthana video-to-motion model"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video": ("VIDEO", {}),
                "motion_name": ("STRING", {"default": "", "multiline": False}),
                "model": (["video-to-motion-v2"], {"default": "video-to-motion-v2"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("motion_id",)
    FUNCTION = "execute"
    CATEGORY = "Uthana"

    @staticmethod
    def path_from_video(video: object) -> tuple[str, str | None]:
        """Resolve a Comfy Video input to a filesystem path for upload. Returns (path, tmp_to_delete)."""
        src = video.get_stream_source()
        if isinstance(src, str):
            p = os.path.abspath(os.path.expanduser(src))
            if not os.path.isfile(p):
                raise RuntimeError(f"Video file not found: {p}")
            return (p, None)
        if isinstance(src, io.BytesIO):
            src.seek(0)
            data = src.read()
            ext = "mp4"
            try:
                if hasattr(video, "get_container_format"):
                    name = (video.get_container_format() or "mp4").lower()
                    if name in ("mov", "avi", "mp4"):
                        ext = name
            except Exception:
                pass
            with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tf:
                tf.write(data)
                tmp = tf.name
            return (tmp, tmp)
        raise RuntimeError(f"Unsupported video source type: {type(src)!r}")

    @staticmethod
    def parse_video_to_motion_result(result: object) -> str:
        """Get motion_id from a finished create_video_to_motion job result."""
        if result is None:
            raise RuntimeError("Video-to-motion job finished but result is empty.")
        if isinstance(result, str):
            result = json.loads(result)
        if not isinstance(result, dict):
            raise RuntimeError(f"Unexpected job result type: {type(result)!r}")

        payload = result.get("result") if isinstance(result.get("result"), dict) else result
        motion_id = payload.get("motion_id") or payload.get("id")
        if not motion_id:
            raise RuntimeError(f"Could not parse motion id from job result: {result!r}")
        return str(motion_id)

    async def execute(
        self,
        video: object,
        motion_name: str,
        model: str,
    ):
        try:
            import uthana
        except ImportError:
            raise ImportError("This node requires uthana. Install with: pip install uthana")

        client = uthana.Client(get_api_key())
        path: str
        tmp_to_remove: str | None = None
        if video is None:
            raise RuntimeError("Connect Load Video (video) to a file on disk.")
        path, tmp_to_remove = self.path_from_video(video)
        name = (motion_name or "").strip() or None

        try:
            job = await client.vtm.create(path, motion_name=name, model=model)
            job_id = job.get("id")
            if not job_id:
                raise RuntimeError(f"Video-to-motion create did not return a job id: {job!r}")
            while str(job.get("status") or "").upper() not in ("FINISHED", "FAILED"):
                await asyncio.sleep(5.0)
                job = await client.jobs.get(job["id"])
            if str(job.get("status") or "").upper() == "FINISHED":
                if job.get("result") is None:
                    job = await client.jobs.get(job_id)
                mid = self.parse_video_to_motion_result(job.get("result"))
                return (mid,)
            if str(job.get("status") or "").upper() == "FAILED":
                raise RuntimeError(
                    f"Video-to-motion job failed (job_id={job_id!r}, status={job.get('status')!r}, result={job.get('result')!r})"
                )
        finally:
            if tmp_to_remove and os.path.isfile(tmp_to_remove):
                try:
                    os.unlink(tmp_to_remove)
                except OSError:
                    pass


class DownloadMotion:
    """Download a motion file from Uthana API"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "character_id": ("STRING", {"default": "c7uJKSavZ5aC"}),
                "motion_id": ("STRING", {"default": ""}),
                "output_format": (OUTPUT_FORMATS, {"default": "GLB"}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120, "step": 1}),
                "no_mesh": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "Uthana"

    async def execute(self, character_id, motion_id, output_format, fps, no_mesh):
        try:
            import uthana
        except ImportError:
            raise ImportError("This node requires uthana. Install with: pip install uthana")

        client = uthana.Client(get_api_key())
        ext = output_format.lower()
        content = await client.motions.download(
            character_id,
            motion_id,
            output_format=ext,
            fps=fps,
            no_mesh=no_mesh,
        )

        output_dir = folder_paths.get_output_directory()
        filename = f"{character_id}-{motion_id}.{ext}"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "wb") as f:
            f.write(content)

        return (filepath,)


class CreateCharacter:
    """Auto-rig a 3D mesh using Uthana API"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "file_path": ("STRING", {"default": ""}),
                "auto_rig": ("BOOLEAN", {"default": True}),
                "front_facing": ("BOOLEAN", {"default": True}),
                "rerig_target": ("STRING", {"default": ""}),
                "include_fingers": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("character_id", "url", "auto_rig_confidence")
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "Uthana"

    async def execute(self, file_path, auto_rig, front_facing, rerig_target, include_fingers):
        if not os.path.isabs(file_path):
            for d in [folder_paths.get_output_directory(), folder_paths.get_input_directory()]:
                candidate = os.path.join(d, file_path)
                if os.path.isfile(candidate):
                    file_path = candidate
                    break

        try:
            import uthana
        except ImportError:
            raise ImportError("This node requires uthana. Install with: pip install uthana")

        client = uthana.Client(get_api_key())
        rerig_target = (rerig_target or "").strip() or None
        char_output = await client.characters.create_from_file(file_path, auto_rig=auto_rig, front_facing=front_facing, rerig_target=rerig_target, include_fingers=include_fingers)

        return (char_output.character_id, char_output.url, char_output.auto_rig_confidence)


class DownloadCharacter:
    """Download a rigged character from Uthana API"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "character_id": ("STRING", {"default": ""}),
                "output_format": (OUTPUT_FORMATS, {"default": "GLB"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "Uthana"

    async def execute(self, character_id, output_format):
        try:
            import uthana
        except ImportError:
            raise ImportError("This node requires uthana. Install with: pip install uthana")

        client = uthana.Client(get_api_key())
        ext = output_format.lower()
        content = await client.characters.download(character_id, output_format=ext)

        output_dir = folder_paths.get_output_directory()
        filename = f"{character_id}-character.{ext}"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "wb") as f:
            f.write(content)

        return (filepath,)


NODE_CLASS_MAPPINGS = {
    "TextToMotion": TextToMotion,
    "DownloadMotion": DownloadMotion,
    "CreateCharacter": CreateCharacter,
    "DownloadCharacter": DownloadCharacter,
    "VideoToMotion": VideoToMotion,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextToMotion": "Text to Motion",
    "DownloadMotion": "Download Motion",
    "CreateCharacter": "Create Character",
    "DownloadCharacter": "Download Character",
    "VideoToMotion": "Video to Motion",
}
