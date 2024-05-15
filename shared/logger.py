import logging
from colorama import init, Fore, Style

init(autoreset=True)  # Initialize colorama


# Custom format with colors for the console
class CustomFormatter(logging.Formatter):
    grey = Fore.WHITE
    yellow = Fore.YELLOW
    red = Fore.RED
    bold_red = Style.BRIGHT + Fore.RED
    reset = Style.RESET_ALL
    format = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    )

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


logger = logging.getLogger("MAIN")
logger.setLevel(logging.DEBUG)

# console handler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)

# file handler
fh = logging.FileHandler("main.log", encoding="utf-8")
fh.setLevel(logging.DEBUG)
log_format = (
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
)
file_formatter = logging.Formatter(log_format)
fh.setFormatter(file_formatter)
logger.addHandler(fh)
