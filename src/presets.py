from src.configurations import Circle, Rectangle, Configuration

# Width of the bounding box:
# TODO: scale and translate positions to coordinates of bounding box
offset = 0.0234375  # n_grid = 3, quality = 1
# offset = 0.0078125 # n_grid = 1, quality = 1

configuration_list = [
    Configuration(
        name="Waterspout Hits Body of Water [Water]",
        geometries=[
            Rectangle(
                lower_left=(0.0, 0.0),
                size=(1.0, 0.1),
                velocity=(0, 0),
            ),
            *[
                Rectangle(
                    size=(0.04, 0.04),
                    velocity=(0, -3),
                    lower_left=(0.48, 0.68),
                    frame_threshold=i,
                )
                for i in range(1, 300)
            ],
        ],
    ),
    Configuration(
        name="Dam Break [Water]",
        geometries=[
            Rectangle(
                lower_left=(offset, offset),
                size=(0.5 - offset, 0.5 - offset),
                velocity=(0, 0),
            ),
        ],
    ),
    Configuration(
        name="Simple Spout Source [Water]",
        geometries=[
            Rectangle(
                lower_left=(0.48, 0.68),
                velocity=(0, -3),
                size=(0.04, 0.04),
                frame_threshold=i,
            )
            for i in range(1, 300)
        ],
    ),
    Configuration(
        name="Spherefall [Water]",
        geometries=[Circle(center=(0.5, 0.35), velocity=(0, -2), radius=0.08)],
    ),
    Configuration(
        name="Stationary Pool [Water]",
        geometries=[
            Rectangle(
                lower_left=(offset, offset),
                size=(1.0, 0.25),
                velocity=(0, 0),
            ),
        ],
    ),
]

# Sort by length in descending order:
configuration_list.sort(key=lambda c: len(c.name), reverse=True)
