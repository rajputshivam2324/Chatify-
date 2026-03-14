from typing import Any

from huggingface_hub import InferenceClient

from app.core.config import get_settings

settings = get_settings()

_image_client: InferenceClient | None = None


def _get_image_client() -> InferenceClient:
    global _image_client
    if _image_client is None:
        _image_client = InferenceClient(token=settings.hf_token)
    return _image_client


async def generate_image(prompt: str) -> bytes:
    """
    Calls stabilityai/stable-diffusion-xl-base-1.0.
    Returns PNG bytes.
    """
    from io import BytesIO

    client = _get_image_client()

    try:
        image = client.text_to_image(
            prompt,
            model="stabilityai/stable-diffusion-xl-base-1.0",
        )
        # Convert PIL Image to bytes
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    except Exception as e:
        raise RuntimeError(f"Image generation failed: {str(e)}")


async def edit_image(image_data: bytes, prompt: str) -> bytes:
    """
    Calls nvidia/ChronoEdit-14B-Diffusers via fal-ai.
    Accepts raw bytes, returns PNG bytes.
    """
    import httpx

    url = "https://queue.fal.run/fal-ai/chronoediting"

    async with httpx.AsyncClient() as client:
        files = {"image": ("image.png", image_data, "image/png")}
        data = {"prompt": prompt}

        response = await client.post(
            url,
            files=files,
            data=data,
            headers={"Authorization": f"Bearer {settings.hf_token}"},
            timeout=120.0,
        )

        if response.status_code != 200:
            raise RuntimeError(f"Image edit failed: {response.text}")

        result = response.json()
        image_url = result.get("images", [{}])[0].get("url")

        if not image_url:
            raise RuntimeError("No image URL returned from ChronoEdit")

        image_response = await client.get(image_url)
        return image_response.content