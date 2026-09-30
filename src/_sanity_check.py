from src.paths import ensure_dirs, PROJECT_ROOT, RAW_DIR
from src.utils.logger import get_logger

def main():
    ensure_dirs()
    logger = get_logger("sanity")
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Raw data dir: {RAW_DIR}")

if __name__ == "__main__":
    main()