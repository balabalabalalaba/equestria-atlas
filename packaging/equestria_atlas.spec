# PyInstaller spec for "Equestria Atlas" (the explorable game). Build with packaging/build_mac.sh or
# packaging\build_windows.bat. The game's sources ship as data files (they are imported at run time by
# name, e.g. game/regions/<seg>.py and scenes/sNN_*.py); every module they import is listed as a
# hidden import, found by scanning their import lines.
import ast
import glob
import os
import sys

ROOT = os.path.abspath(os.path.join(SPECPATH, '..'))
SRC = ('engine', 'game', 'scenes', 'audio')
OWN = set(SRC)


def own_sources():
    for d in SRC:
        for fn in glob.glob(os.path.join(ROOT, d, '**', '*.py'), recursive=True):
            yield fn


def imported_modules():
    mods = set()
    for fn in own_sources():
        try:
            tree = ast.parse(open(fn, encoding='utf-8').read())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module] + [node.module + '.' + a.name for a in node.names]
            else:
                continue
            for n in names:
                if n.split('.')[0] not in OWN:
                    mods.add(n)
    return sorted(mods)


datas = []
for fn in own_sources():
    rel = os.path.relpath(os.path.dirname(fn), ROOT)
    datas.append((fn, rel))
datas += [(os.path.join(ROOT, 'audio', 'out', 'soundtrack.wav'), 'audio/out'),
          (os.path.join(ROOT, 'audio', 'out', 'sections.json'), 'audio/out')]
# the game's own soundtrack (audio/game_music.py): the OGG loops and their list
datas += [(fn, 'audio/out/game') for fn in glob.glob(os.path.join(ROOT, 'audio', 'out', 'game', '*.ogg'))]
datas += [(os.path.join(ROOT, 'audio', 'out', 'game', 'tracks.json'), 'audio/out/game')]
datas += [(fn, 'out/cache') for fn in glob.glob(os.path.join(ROOT, 'out', 'cache', 'worldmap_*.npz'))]

def is_module(name):
    import importlib.util
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, AttributeError):
        return False


hidden = [m for m in imported_modules() if m.split('.')[0] not in ('tools',) and is_module(m)]
hidden += ['pygame._sdl2.video', 'scipy.io.wavfile', 'scipy.ndimage', 'scipy.interpolate', 'scipy.spatial']

a = Analysis([os.path.join(ROOT, 'packaging', 'launch.py')], pathex=[], datas=datas, hiddenimports=hidden,
             excludes=['tkinter', 'matplotlib', 'IPython', 'PyInstaller'], noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='Equestria Atlas', console=False,
          icon=os.path.join(ROOT, 'packaging', 'icon.icns' if sys.platform == 'darwin' else 'icon.ico'))
coll = COLLECT(exe, a.binaries, a.datas, name='Equestria Atlas')
if sys.platform == 'darwin':
    app = BUNDLE(coll, name='Equestria Atlas.app', icon=os.path.join(ROOT, 'packaging', 'icon.icns'),
                 bundle_identifier='nz.equestria.atlas',
                 info_plist={'NSHighResolutionCapable': True, 'CFBundleShortVersionString': '1.0'})
