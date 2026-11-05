"""App icon: the explorer's balloon on a rounded sky tile, pixel-scaled. Writes packaging/icon.png,
icon.ico (Windows) and, on a Mac, icon.icns (via iconutil)."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from PIL import Image

from engine import Canvas
from game.compose import balloon_rgba

HERE = os.path.join(ROOT, 'packaging')


def tile():
    c = Canvas(32, 32, color='#4d9be6')
    c.gradient(0, 0, 32, 32, ['#4d9be6', '#86cbfb', '#ffd6e8'], soft=0.4)
    c.ellipse(8, 26, 7, 3, 'white')
    c.ellipse(25, 23, 6, 2, 'white')
    a = np.zeros((32, 32, 4), np.uint8)
    a[..., :3] = c.a
    a[..., 3] = 255
    for (x, y) in ((0, 0), (31, 0), (0, 31), (31, 31), (1, 0), (0, 1), (30, 0), (31, 1), (0, 30), (1, 31),
                   (31, 30), (30, 31)):
        a[y, x, 3] = 0
    b = balloon_rgba('l')
    h, w = b.shape[:2]
    x0, y0 = (32 - w) // 2, (32 - h) // 2
    m = b[..., 3] > 127
    a[y0:y0 + h, x0:x0 + w][m] = b[m]
    return a


def main():
    a = tile()
    big = Image.fromarray(a, 'RGBA').resize((512, 512), Image.NEAREST)
    big.save(os.path.join(HERE, 'icon.png'))
    big.save(os.path.join(HERE, 'icon.ico'), sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    if sys.platform == 'darwin':
        d = os.path.join(HERE, 'icon.iconset')
        os.makedirs(d, exist_ok=True)
        for s in (16, 32, 128, 256, 512):
            Image.fromarray(a, 'RGBA').resize((s, s), Image.NEAREST).save(os.path.join(d, f'icon_{s}x{s}.png'))
            Image.fromarray(a, 'RGBA').resize((2 * s, 2 * s), Image.NEAREST).save(os.path.join(d, f'icon_{s}x{s}@2x.png'))
        subprocess.run(['iconutil', '-c', 'icns', d, '-o', os.path.join(HERE, 'icon.icns')], check=True)
        for fn in os.listdir(d):
            os.remove(os.path.join(d, fn))
        os.rmdir(d)
    print('icons written to', HERE)


if __name__ == '__main__':
    main()
