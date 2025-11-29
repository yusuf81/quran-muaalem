"""Model manager for lazy loading ML models with proper error handling."""

import logging
from typing import Optional, Tuple
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioFrameClassification

from ..inference import Muaalem
from ..exceptions import ModelLoadError

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages lazy loading of ML models."""
    
    def __init__(
        self, 
        device: str = "cpu",
        muaalem_model_id: str = "obadx/muaalem-model-v3_2",
        segmenter_model_id: str = "obadx/recitation-segmenter-v2"
    ):
        """Initialize model manager.
        
        Args:
            device: Device to load models on ('cpu' or 'cuda')
            muaalem_model_id: HuggingFace model ID for Muaalem
            segmenter_model_id: HuggingFace model ID for segmenter
        """
        self.device = device
        self.muaalem_model_id = muaalem_model_id
        self.segmenter_model_id = segmenter_model_id
        
        # Private model storage
        self._muaalem: Optional[Muaalem] = None
        self._segmenter_model = None
        self._segmenter_processor = None
        
        logger.info(f"ModelManager initialized with device: {device}")
    
    @property
    def muaalem(self) -> Muaalem:
        """Get Muaalem model, loading lazily if needed.
        
        Returns:
            Muaalem model instance
            
        Raises:
            ModelLoadError: If model fails to load
        """
        if self._muaalem is None:
            try:
                logger.info(f"Loading Muaalem model from {self.muaalem_model_id}...")
                self._muaalem = Muaalem(
                    model_name_or_path=self.muaalem_model_id,
                    device=self.device
                )
                logger.info("Muaalem model loaded successfully")
            except Exception as e:
                error_msg = f"Failed to load Muaalem model: {str(e)}"
                logger.error(error_msg, exc_info=True)
                raise ModelLoadError(error_msg) from e
        
        return self._muaalem
    
    @property
    def segmenter(self) -> Tuple[AutoModelForAudioFrameClassification, AutoFeatureExtractor]:
        """Get segmenter model and processor, loading lazily if needed.
        
        Returns:
            Tuple of (model, processor)
            
        Raises:
            ModelLoadError: If model fails to load
        """
        if self._segmenter_model is None:
            try:
                logger.info(f"Loading Recitation Segmenter from {self.segmenter_model_id}...")
                
                self._segmenter_processor = AutoFeatureExtractor.from_pretrained(
                    self.segmenter_model_id
                )
                self._segmenter_model = AutoModelForAudioFrameClassification.from_pretrained(
                    self.segmenter_model_id
                )
                self._segmenter_model.to(self.device, dtype=torch.bfloat16)
                
                logger.info("Recitation Segmenter loaded successfully")
            except Exception as e:
                error_msg = f"Failed to load Segmenter model: {str(e)}"
                logger.error(error_msg, exc_info=True)
                raise ModelLoadError(error_msg) from e
        
        return self._segmenter_model, self._segmenter_processor
    
    def is_muaalem_loaded(self) -> bool:
        """Check if Muaalem model is loaded.
        
        Returns:
            True if loaded, False otherwise
        """
        return self._muaalem is not None
    
    def is_segmenter_loaded(self) -> bool:
        """Check if Segmenter model is loaded.
        
        Returns:
            True if loaded, False otherwise
        """
        return self._segmenter_model is not None
    
    def unload_muaalem(self):
        """Unload Muaalem model to free memory."""
        if self._muaalem is not None:
            logger.info("Unloading Muaalem model")
            del self._muaalem
            self._muaalem = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def unload_segmenter(self):
        """Unload Segmenter model to free memory."""
        if self._segmenter_model is not None:
            logger.info("Unloading Segmenter model")
            del self._segmenter_model
            del self._segmenter_processor
            self._segmenter_model = None
            self._segmenter_processor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def unload_all(self):
        """Unload all models to free memory."""
        self.unload_muaalem()
        self.unload_segmenter()
        logger.info("All models unloaded")
    
    def get_status(self) -> dict:
        """Get loading status of all models.
        
        Returns:
            Dictionary with model loading status
        """
        return {
            "muaalem_loaded": self.is_muaalem_loaded(),
            "segmenter_loaded": self.is_segmenter_loaded(),
            "device": self.device
        }
