import os
import logging

def setup_logging(config, log_prefix):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set to lowest level to capture all logs
    
    output_dir = config['logging'].get('output_dir', './')
    os.makedirs(output_dir, exist_ok=True)
    
    log_levels = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
    }

    handlers = {}
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    for level_name, level in log_levels.items():
        log_file = os.path.join(output_dir, f'{log_prefix}_{level_name}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        handlers[level_name] = file_handler

    return handlers

def cleanup_logging(logger, handlers):
    for handler in handlers.values():
        logger.removeHandler(handler)
        handler.close()