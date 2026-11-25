import cloudinary
import cloudinary.uploader
from app.core.config import settings

cloudinary.config(
    cloud_name=settings.CLOUDINARY_CLOUD_NAME,
    api_key=settings.CLOUDINARY_API_KEY,
    api_secret=settings.CLOUDINARY_API_SECRET
)

def upload_image_to_cloudinary(file_content: bytes, filename: str, user_id: str = "anonymous") -> str:
    """Uploads an image to Cloudinary and returns the secure URL."""
    folder_path = f"pet-breed-classifier/users/{user_id}"
    
    response = cloudinary.uploader.upload(
        file_content,
        folder=folder_path,
        resource_type="image"
    )
    return response.get("secure_url")
