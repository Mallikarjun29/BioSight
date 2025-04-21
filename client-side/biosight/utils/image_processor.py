"""Image processing and prediction utilities."""
import torch
from PIL import Image
from torchvision import transforms
from fastapi import HTTPException
import logging
from .config import IMAGE_TRANSFORMS, MODEL_SETTINGS
from .constants import CLASS_NAMES, STATUS_MESSAGES

logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_TRANSFORMS['resize']),
            transforms.CenterCrop(IMAGE_TRANSFORMS['crop_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_TRANSFORMS['mean'], std=IMAGE_TRANSFORMS['std']),
        ])

    def predict(self, image_path: str) -> str:
        """
        Predict the class of an image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            str: Predicted class name
        """
        if self.model is None:
            raise ValueError(STATUS_MESSAGES['MODEL_LOAD_ERROR'])
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0)
            image_tensor = image_tensor.to(self.device)

            # Make prediction
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                max_prob, predicted_idx = torch.max(probabilities, 1)

            # Get class name based on confidence
            if max_prob.item() >= MODEL_SETTINGS['confidence_threshold']:
                idx = predicted_idx.item()
                if 0 <= idx < len(CLASS_NAMES):
                    return CLASS_NAMES[idx]
                return "unknown_index"
            return "unknown"

        except Exception as e:
            logger.error(f"Error predicting image {image_path}: {e}")
            raise HTTPException(
                status_code=500,
                detail=STATUS_MESSAGES['PREDICTION_ERROR'].format(str(e))
            )