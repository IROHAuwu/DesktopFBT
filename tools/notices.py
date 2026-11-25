"""建置時執行：把「這次建置實際安裝的套件」的授權檔原文收進 THIRD_PARTY_NOTICES/_build，
並寫出 BUILD.md（套件、版本、授權宣告、收到哪些檔案）。

THIRD_PARTY_NOTICES/INDEX.md 是人工整理的說明；CI 每次安裝到的版本可能不同，
這一份保證發布出去的 EXE 附的授權檔和裡面的套件是同一版。

用法：python tools/notices.py [輸出資料夾，預設 THIRD_PARTY_NOTICES]
"""
import datetime
import os
import re
import shutil
import sys
from importlib import metadata

# 只在建置時用到、不會進 EXE 的工具（PyInstaller 的 bootloader 另外列在 INDEX.md）
BUILD_ONLY = {'pip', 'setuptools', 'wheel', 'pyinstaller', 'pyinstaller-hooks-contrib', 'altgraph',
              'pefile', 'pywin32-ctypes', 'macholib'}
WANTED = re.compile(r'(^|[\\/])(licen[cs]e|copying|notice|authors|third.?party)[^\\/]*$', re.I)


def norm(name):
    return re.sub(r'[-_.]+', '-', name or '').lower()


def license_of(md):
    exp = md.get('License-Expression')
    if exp:
        return exp
    lic = (md.get('License') or '').strip()
    if lic and '\n' not in lic and len(lic) < 80:
        return lic
    cls = [c.split('::')[-1].strip() for c in md.get_all('Classifier') or [] if c.startswith('License ::')]
    if cls:
        return ' / '.join(cls)
    return '(see the files)' if lic else 'UNKNOWN'


def collect(out):
    dest = os.path.join(out, '_build')
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest)
    rows, seen = [], set()
    for d in sorted(metadata.distributions(), key=lambda d: norm(d.metadata.get('Name'))):
        name = d.metadata.get('Name')
        if not name or norm(name) in BUILD_ONLY or norm(name) in seen:
            continue
        seen.add(norm(name))
        folder = os.path.join(dest, f'{norm(name)}-{d.version}')
        files = []
        for f in d.files or []:
            s = str(f)
            if '.dist-info' not in s and not s.count('/') <= 1:
                continue                      # 只收 dist-info 裡的，以及套件最上層的授權檔
            if not WANTED.search(s) or s.endswith(('.py', '.pyc')):
                continue
            src = d.locate_file(f)
            if not os.path.isfile(src):
                continue
            rel = s.split('.dist-info/', 1)[-1].replace('\\', '/')
            dst = os.path.join(folder, *rel.split('/'))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)
            files.append(rel)
        lic = (d.metadata.get('License') or '').strip()
        if not files and lic and ('\n' in lic or len(lic) >= 80):
            os.makedirs(folder, exist_ok=True)       # 授權原文只寫在 METADATA 的 License 欄位裡
            with open(os.path.join(folder, 'METADATA_License_field.txt'), 'w', encoding='utf-8') as f:
                f.write(lic + '\n')
            files.append('METADATA_License_field.txt')
        home = d.metadata.get('Home-page') or next(
            (u.split(',', 1)[1].strip() for u in d.metadata.get_all('Project-URL') or [] if ',' in u), '')
        rows.append((name, d.version, license_of(d.metadata), files, home))
    with open(os.path.join(out, 'BUILD.md'), 'w', encoding='utf-8') as f:
        f.write('# Packages installed for this build\n\n')
        f.write(f'Generated {datetime.date.today().isoformat()} by `tools/notices.py` from the build '
                'environment (Python %d.%d.%d). ' % sys.version_info[:3])
        f.write('License files are copied verbatim into `_build/<package>-<version>/`. '
                'See INDEX.md for the runtime, Microsoft binaries and models.\n\n')
        f.write('| Package | Version | Declared license | Files | Source |\n|---|---|---|---|---|\n')
        for name, ver, lic, files, home in rows:
            shown = ', '.join(files) if files else '**none shipped by the package**'
            f.write(f'| {name} | {ver} | {lic} | {shown} | {home} |\n')
    return rows


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'THIRD_PARTY_NOTICES'
    rows = collect(out)
    missing = [r[0] for r in rows if not r[3]]
    print(f'{len(rows)} packages; no license file shipped: {", ".join(missing) or "none"}')
