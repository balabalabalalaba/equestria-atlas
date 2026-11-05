"""One module per explorable region, named after its segment id in engine/timeline.py
(game/regions/<seg>.py, e.g. ponyville.py, sweetapple.py). Each defines one `Region` subclass
(game/region.py). game/world.py discovers them by file name and imports them on first landing."""
