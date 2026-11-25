"""デスクトップFBT 的圖示：螢幕（桌面）前面一個由關節點連成的人（全身追蹤）。python make_icon.py <輸出資料夾>"""
import sys

from PIL import Image, ImageDraw

BG_TOP, BG_BOTTOM = (38, 41, 58), (22, 24, 31)
SCREEN = (88, 94, 122)
BONE = (139, 124, 255)            # 介面的主色 #8b7cff
JOINT = (240, 241, 246)

# 1024 座標的骨架：一手舉起、跨一步（看得出在動）
P = dict(head=(512, 262), neck=(512, 372), hip=(512, 580), ls=(410, 392), rs=(614, 392),
         le=(330, 300), lw=(318, 180), re=(700, 480), rw=(772, 392),
         lk=(424, 712), la=(372, 852), rk=(612, 704), ra=(700, 836))
BONES = [('neck', 'hip'), ('ls', 'rs'), ('ls', 'le'), ('le', 'lw'), ('rs', 're'), ('re', 'rw'),
         ('hip', 'lk'), ('lk', 'la'), ('hip', 'rk'), ('rk', 'ra')]


def draw(size, detail=True):
    S = 4 if size >= 64 else 8                      # 超取樣
    n = size * S
    k = n / 1024.0
    img = Image.new('RGBA', (n, n), (0, 0, 0, 0))
    grad = Image.new('RGBA', (n, n))
    gd = ImageDraw.Draw(grad)
    for y in range(n):
        t = y / max(n - 1, 1)
        gd.line([(0, y), (n, y)], fill=tuple(int(a + (b - a) * t) for a, b in zip(BG_TOP, BG_BOTTOM)) + (255,))
    mask = Image.new('L', (n, n), 0)
    m = 36 * k if detail else 0
    ImageDraw.Draw(mask).rounded_rectangle([m, m, n - m, n - m], radius=(210 if detail else 190) * k, fill=255)
    img.paste(grad, (0, 0), mask)
    d = ImageDraw.Draw(img)
    grow = 1.0 if detail else 1.14                  # 小尺寸：只畫人，放大、線加粗

    def pt(name):
        x, y = P[name]
        return ((x - 512) * grow + 512) * k, ((y - 520) * grow + 520) * k
    if detail:                                      # 螢幕（在人的後面）
        w = 34 * k
        d.rounded_rectangle([150 * k, 330 * k, 874 * k, 770 * k], radius=46 * k, outline=SCREEN, width=int(w))
        d.rounded_rectangle([452 * k, 770 * k, 572 * k, 850 * k], radius=0, fill=SCREEN)
        d.rounded_rectangle([372 * k, 842 * k, 652 * k, 876 * k], radius=17 * k, fill=SCREEN)
    bw = (54 if detail else 84) * k
    for a, b in BONES:
        (x0, y0), (x1, y1) = pt(a), pt(b)
        d.line([x0, y0, x1, y1], fill=BONE, width=int(bw))
        for x, y in ((x0, y0), (x1, y1)):
            d.ellipse([x - bw / 2, y - bw / 2, x + bw / 2, y + bw / 2], fill=BONE)
    hx, hy = pt('head')
    hr = (74 if detail else 92) * k
    d.ellipse([hx - hr, hy - hr, hx + hr, hy + hr], fill=BONE)
    if detail:
        jr = 27 * k
        for name in P:
            if name == 'head':
                continue
            x, y = pt(name)
            d.ellipse([x - jr, y - jr, x + jr, y + jr], fill=JOINT)
        d.ellipse([hx - 30 * k, hy - 30 * k, hx + 30 * k, hy + 30 * k], fill=JOINT)
    return img.resize((size, size), Image.LANCZOS)


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else '.'
    draw(1024).save(f'{out}/icon.png')
    sizes = [16, 20, 24, 32, 40, 48, 64, 128, 256]
    frames = [draw(s, detail=s >= 48) for s in sizes]
    frames[-1].save(f'{out}/icon.ico', format='ICO', append_images=frames[:-1], sizes=[(s, s) for s in sizes])
    sheet = Image.new('RGBA', (16 + sum(s + 16 for s in sizes), 256 + 32), (200, 200, 205, 255))
    x = 16
    for s, f in zip(sizes, frames):
        sheet.paste(f, (x, 16), f)
        x += s + 16
    sheet.save(f'{out}/preview.png')
