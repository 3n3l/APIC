from taichi import hex_to_rgb


class Classification:
    Colliding = 11
    Interior = 22
    Empty = 33


class Color:
    # IBM Carbon
    Background = hex_to_rgb(0x007d79) # teal 60
    Water = hex_to_rgb(0x78a9ff) # blue 40
    Ice = hex_to_rgb(0xd0e2ff) # blue 20

class State:
    Active = 0
    Hidden = 1

GRAVITY = -9.81
