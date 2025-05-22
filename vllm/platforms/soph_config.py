import os
import ctypes as ct

SIMULATE_RANK_NUM = int(os.getenv("SIMULATE_RANK_NUM", "1"))
SLOG_LEVEL = os.getenv("SLOG_LEVEL", 'INFO')

USE_SOPHTPU = os.getenv("DEVICE", '').upper() != "GPU"
KVCACHE_BLOCKS = int(os.getenv('KVCACHE_BLOCKS', '1024'))

# Some global variables about DEBUG
DEBUG_MODE = os.getenv("DEBUG_MODE", "OFF").upper() == "ON"
DECODE_TOKEN_LEN = int(os.getenv("DECODE_TOKEN_LEN", "1024"))
DEBUG_HIDDEN_LAYERS = int(os.getenv("DEBUG_HIDDEN_LAYERS", "1"))
TENSOR_DUMP = os.getenv("TENSOR_DUMP", "OFF").upper() == "ON"
TENSOR_DUMP_PATH = os.getenv("TENSOR_DUMP_PATH", "/workspace/dumped_tensors/")

CONTEXT_LEN = int(os.getenv("CONTEXT_LEN", "6"))
MAX_IMG_TOKEN = int(os.getenv("MAX_IMG_TOKEN", "3000")) # max token num of image part for vlms, used for calulating kv cache blocks
ENABLE_PROFILE = int(os.getenv("ENABLE_PROFILE", "0"))
PROFILE_BOOK_KEEPING = int(os.getenv("PROFILE_BOOK_KEEPING", "1"))
PROFILE_STARTING_TOKEN = int(os.getenv("PROFILE_STARTING_TOKEN", "1"))

MAX_TOTAL_TOKENS = int(os.getenv('MAX_TOTAL_TOKENS', '4046'))

RANK = int(os.getenv("RANK", "0"))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))

if os.getenv("OMPI_COMM_WORLD_RANK"):
    RANK = int(os.getenv("OMPI_COMM_WORLD_RANK"))
if os.getenv("OMPI_COMM_WORLD_SIZE"):
    WORLD_SIZE = int(os.getenv("OMPI_COMM_WORLD_SIZE"))

os.environ['LOCAL_WORLD_SIZE']=str(WORLD_SIZE)
os.environ['LOCAL_RANK']=str(RANK)

# some envs for PLD
SKIP_H2D = os.getenv("SKIP_H2D", "OFF").upper() == "ON"
USE_DUMMY_DATA = os.getenv("USE_DUMMY_DATA", "OFF").upper() == "ON"

BACKBONE_CMD_FORBID = os.getenv("BACKBONE_CMD_FORBID", "OFF").upper() == "ON"
