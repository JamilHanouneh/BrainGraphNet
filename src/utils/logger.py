"""
Logging utilities
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(config, name='BrainGraphNet'):
    """
    Setup logger with file and console handlers
    
    Args:
        config: Configuration dictionary
        name: Logger name
    
    Returns:
        logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    
    # Get logging config
    log_config = config['system']['logging']
    log_level = getattr(logging, log_config['level'])
    logger.setLevel(log_level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    log_dir = Path(log_config['save_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'training_{timestamp}.log'
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    logger.info(f"Logger initialized. Logging to {log_file}")
    
    return logger


def log_system_info(logger):
    """Log system information"""
    import platform
    import torch
    
    logger.info("="*60)
    logger.info("System Information")
    logger.info("="*60)
    logger.info(f"Python Version: {platform.python_version()}")
    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Processor: {platform.processor()}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info("="*60)
