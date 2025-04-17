# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
from PIL import Image, ImageDraw
from matplotlib.offsetbox import OffsetImage
import numpy as np

ROOT_DIR = "/Users/haidark/Library/CloudStorage/GoogleDrive-haidark@gmail.com/My Drive/Zero Sum Eval/final-rankings-3-9-25/"
ALL_DIRS = {
    "chess": "rankings-3-9-25_chess",
    "debate": "rankings-3-9-25_debate-new",
    "gandalf": "rankings-3-9-25_gandalf_final_500",
    "liars_dice": "rankings-3-9-25_liars_dice_reasoning_1000",
    "mathquiz": "rankings-3-9-25_mathquiz_final_500",
    "poker": "rankings-3-9-25_poker_final_500",
    "pyjail": "rankings-3-9-25_pyjail-new"
}

ROLE_WEIGHTS = {
    "chess": {
        "white": 1.0,
        "black": 2.0
    },
    "debate": None,
    "gandalf": {
        "sentinel": 1.0,
        "infiltrator": 2.0
    },
    "liars_dice": None,
    "mathquiz": {
        "student": 1.0,
        "teacher": 2.0
    },
    "poker": None,
    "pyjail": {
        "defender": 2.0,
        "attacker": 1.0
    }
}

# Custom color palette - using different shades of blue
CUSTOM_COLORS = [
    (14/255, 140/255, 247/255),   # Bright blue
    (41/255, 44/255, 147/255),    # Deep blue
    (0/255, 84/255, 159/255),     # Navy blue
    (86/255, 180/255, 233/255),   # Sky blue
    (120/255, 180/255, 210/255),  # Darker light blue for mathquiz
    (0/255, 119/255, 182/255),    # Medium blue
    (65/255, 105/255, 225/255)    # Royal blue
]

# Create a mapping between games and their colors for consistency
GAME_COLOR_MAPPING = {
    "chess": CUSTOM_COLORS[0],
    "debate": CUSTOM_COLORS[1],
    "gandalf": CUSTOM_COLORS[2],
    "liars_dice": CUSTOM_COLORS[3],
    "mathquiz": CUSTOM_COLORS[4],
    "poker": CUSTOM_COLORS[5],
    "pyjail": CUSTOM_COLORS[6]
}

# Map model names to their logo files
LOGO_DIR = "paper/logos"
LOGO_MAPPING = {
    "gpt-4o": os.path.join(LOGO_DIR, "openai.png"),
    "claude-3.7-sonnet": os.path.join(LOGO_DIR, "claude.png"),
    "claude-3.7-sonnet-thinking": os.path.join(LOGO_DIR, "claude.png"),
    "gemini-2.0-flash": os.path.join(LOGO_DIR, "gemini.png"),
    "llama-3.3-70b": os.path.join(LOGO_DIR, "llama.png"),
    "llama-3.1-405b": os.path.join(LOGO_DIR, "llama.png"),
    "llama-3.1-70b": os.path.join(LOGO_DIR, "llama.png"),
    "llama-3.1-8b": os.path.join(LOGO_DIR, "llama.png"),
    "deepseek-chat": os.path.join(LOGO_DIR, "deepseek.png"),
    "deepseek-r1": os.path.join(LOGO_DIR, "deepseek.png"),
    "qwen2.5-32b": os.path.join(LOGO_DIR, "qwen2.png"),
    "qwq-32b": os.path.join(LOGO_DIR, "qwen2.png"),
    "o3-mini-high": os.path.join(LOGO_DIR, "openai.png")
}

# Function to load and resize logo with circular cropping and guaranteed white background
def get_logo(logo_path, size=0.15):
    try:
        # Use PIL directly to load the image
        pil_img = Image.open(logo_path)

        # Force conversion to RGBA first to properly handle all image types
        if pil_img.mode == 'P':  # Palette mode
            pil_img = pil_img.convert('RGBA')
        elif pil_img.mode != 'RGBA':
            pil_img = pil_img.convert('RGBA')

        # Create a square image by cropping
        width, height = pil_img.size
        size_px = min(width, height)

        # Calculate crop box (centered)
        left = (width - size_px) // 2
        top = (height - size_px) // 2
        right = left + size_px
        bottom = top + size_px

        # Crop to square
        square_img = pil_img.crop((left, top, right, bottom))

        # Create a solid white background image
        white_bg = Image.new('RGB', (size_px, size_px), (255, 255, 255))

        # Create a circular mask
        mask = Image.new('L', (size_px, size_px), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, size_px, size_px), fill=255)

        # Paste the logo onto the white background using the mask
        # This is the key step - we're using the RGBA image as the source but pasting onto RGB
        white_bg.paste(square_img, (0, 0), square_img.split()[3])  # Use alpha channel as mask

        # Resize to a standard size
        target_size = (100, 100)
        resized_img = white_bg.resize(target_size, Image.Resampling.LANCZOS)

        # Convert to numpy array for matplotlib
        img_array = np.array(resized_img)

        # Create an OffsetImage with the standardized image
        offset_img = OffsetImage(img_array, zoom=size)

        # Force the image to have a white background in matplotlib
        offset_img.set_zorder(10)  # Ensure it's on top

        return offset_img
    except Exception as e:
        print(f"Error loading logo {logo_path}: {e}")
        # Create a fallback simple circle with the first letter of the model
        model_name = os.path.basename(logo_path).split('.')[0]
        first_letter = model_name[0].upper() if model_name else "?"

        # Create a white circle with text
        fallback = Image.new('RGB', (100, 100), (255, 255, 255))
        draw = ImageDraw.Draw(fallback)
        draw.ellipse((0, 0, 99, 99), outline=(100, 100, 100), width=2)

        # Add text (centered)
        try:
            # Try to use a font if available
            from PIL import ImageFont
            font = ImageFont.truetype("Arial", 40)
            text_width, text_height = draw.textsize(first_letter, font=font)
            draw.text((50-text_width//2, 50-text_height//2), first_letter, fill=(0, 0, 0), font=font)
        except:
            # Fallback if font not available
            draw.text((40, 30), first_letter, fill=(0, 0, 0))

        img_array = np.array(fallback)
        return OffsetImage(img_array, zoom=size)
