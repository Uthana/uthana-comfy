import json
import math
import os
import threading
import time
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

    resolved = getattr(uthana.DefaultCharacters(), value, None)
    if resolved is not None:
        return resolved
    if value.startswith("c"):
        return value
    raise uthana.Error(
        f"Invalid character_id: '{value}'. Use a name ({', '.join(_CHARACTER_NAMES)}) or a character ID starting with 'c'."
    )


class TextToMotionVqvaeV1:
    """Generate motion from text prompt using Uthana text-to-motion vqvae-v1 model"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "character_id": ("STRING", {"default": "tar"}),
                "foot_ik": ("BOOLEAN", {"default": True}),
                "staging": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("character_id", "motion_id")
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "Uthana"

    def execute(self, prompt, character_id, foot_ik, staging):
        try:
            import uthana
        except ImportError:
            raise ImportError("This node requires uthana. Install with: pip install uthana")

        client = uthana.Client(get_api_key(), staging=staging)
        result = run_with_progress(
            lambda: client.create_text_to_motion(
                "vqvae-v1",
                prompt,
                character_id=resolve_character_id(character_id),
                foot_ik=foot_ik,
            ),
            estimated_duration=10.0,
        )

        return (result.character_id, result.motion_id)


class TextToMotionDiffusionV2:
    """Generate motion from text prompt using Uthana text-to-motion diffusion-v2 model with extended controls"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "character_id": ("STRING", {"default": "tar"}),
                "foot_ik": ("BOOLEAN", {"default": True}),
                "motion_length": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01}),
                "cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "step": 1}),
                "internal_ik": ("BOOLEAN", {"default": True}),
                "staging": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("character_id", "motion_id")
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "Uthana"

    def execute(self, prompt, character_id, foot_ik, motion_length, cfg_scale, seed, internal_ik, staging):
        try:
            import uthana
        except ImportError:
            raise ImportError("This node requires uthana. Install with: pip install uthana")

        client = uthana.Client(get_api_key(), staging=staging)
        result = run_with_progress(
            lambda: client.create_text_to_motion(
                "diffusion-v2",
                prompt,
                character_id=resolve_character_id(character_id),
                foot_ik=foot_ik,
                length=motion_length,
                cfg_scale=cfg_scale,
                seed=seed,
                internal_ik=internal_ik,
            ),
            estimated_duration=10.0,
        )

        return (result.character_id, result.motion_id)


class DownloadMotion:
    """Download a motion file from Uthana API"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "character_id": ("STRING", {"default": ""}),
                "motion_id": ("STRING", {"default": ""}),
                "output_format": (OUTPUT_FORMATS, {"default": "GLB"}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120, "step": 1}),
                "no_mesh": ("BOOLEAN", {"default": True}),
                "staging": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "Uthana"

    def execute(self, character_id, motion_id, output_format, fps, no_mesh, staging):
        try:
            import uthana
        except ImportError:
            raise ImportError("This node requires uthana. Install with: pip install uthana")

        client = uthana.Client(get_api_key(), staging=staging)
        ext = output_format.lower()
        content = client.download_motion(
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
                "staging": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("character_id", "url", "auto_rig_confidence")
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "Uthana"

    def execute(self, file_path, auto_rig, front_facing, staging):
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

        client = uthana.Client(get_api_key(), staging=staging)
        char_output = run_with_progress(
            lambda: client.create_character(file_path, auto_rig=auto_rig, front_facing=front_facing),
            estimated_duration=60.0,
        )

        return (char_output.character_id, char_output.url, char_output.auto_rig_confidence)


class DownloadCharacter:
    """Download a rigged character from Uthana API"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "character_id": ("STRING", {"default": ""}),
                "output_format": (OUTPUT_FORMATS, {"default": "GLB"}),
                "staging": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "Uthana"

    def execute(self, character_id, output_format, staging):
        try:
            import uthana
        except ImportError:
            raise ImportError("This node requires uthana. Install with: pip install uthana")

        client = uthana.Client(get_api_key(), staging=staging)
        ext = output_format.lower()
        content = client.download_character(character_id, output_format=ext)

        output_dir = folder_paths.get_output_directory()
        filename = f"{character_id}-character.{ext}"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "wb") as f:
            f.write(content)

        return (filepath,)


NODE_CLASS_MAPPINGS = {
    "TextToMotionVqvaeV1": TextToMotionVqvaeV1,
    "TextToMotionDiffusionV2": TextToMotionDiffusionV2,
    "DownloadMotion": DownloadMotion,
    "CreateCharacter": CreateCharacter,
    "DownloadCharacter": DownloadCharacter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextToMotionVqvaeV1": "Text to Motion VQVAE v1",
    "TextToMotionDiffusionV2": "Text to Motion Diffusion v2",
    "DownloadMotion": "Download Motion",
    "CreateCharacter": "Create Character",
    "DownloadCharacter": "Download Character",
}
