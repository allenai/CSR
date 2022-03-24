NORMALIZE_RGB_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_RGB_STD = (0.229, 0.224, 0.225)
DEFAULT_NUM_WORKERS = 8
COLOR_JITTER_BRIGHTNESS = 0.4
COLOR_JITTER_CONTRAST = 0.4
COLOR_JITTER_SATURATION = 0.4
COLOR_JITTER_HUE = 0.2
GRAYSCALE_PROBABILITY = 0.2
ROTATIONS = (0., 90., 180., 270.)

# splits from ai2 rearrangement: https://github.com/allenai/ai2thor-rearrangement#-datasets
TRAIN_ROOM_IDS = tuple(list(range(1, 21)) + list(range(201, 221)) +
                       list(range(301, 321)) + list(range(401, 421)))
VAL_ROOM_IDS = tuple(list(range(21, 26)) + list(range(221, 226)) +
                     list(range(321, 326)) + list(range(421, 426)))
TEST_ROOM_IDS = tuple(list(range(26, 31)) + list(range(226, 231)) +
                      list(range(326, 331)) + list(range(426, 431)))
ALL_ROOM_IDS = tuple(list(TRAIN_ROOM_IDS) + list(VAL_ROOM_IDS) + list(TEST_ROOM_IDS))
TRAIN_VAL_ROOM_IDS = tuple(list(TRAIN_ROOM_IDS) + list(VAL_ROOM_IDS))

KITCHEN_FLOOR_PLANS = {f"FloorPlan{i}" for i in range(1, 31)}
LIVING_ROOM_FLOOR_PLANS = {f"FloorPlan{200 + i}" for i in range(1, 31)}
BEDROOM_FLOOR_PLANS = {f"FloorPlan{300 + i}" for i in range(1, 31)}
BATHROOM_FLOOR_PLANS = {f"FloorPlan{400 + i}" for i in range(1, 31)}

CLASSES_TO_IGNORE = (
    'Floor',
    'StoveKnob',
    'StoveBurner',
    'Window',
    'Apple',
    'Bread',
    'Cloth',
    'HandTowel',
    'KeyChain',
    'Lettuce',
    'Pillow',
    'Potato',
    'Tomato',
    'Mirror')

DATALOADER_BOX_FRAC_THRESHOLD = 0.008
RENDERING_BOX_FRAC_THRESHOLD = 0.0016
IMAGE_SIZE = 224
