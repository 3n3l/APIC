from src.configurations import Circle, Rectangle, Configuration

# Width of the bounding box:
# TODO: scale and translate positions to coordinates of bounding box
offset = 0.0234375  # n_grid = 3, quality = 1
# offset = 0.0078125 # n_grid = 1, quality = 1

configuration_list = [
    Configuration(
        name="Waterjet Hits Pool",
        geometries=[
            Rectangle(
                lower_left=(0.0, 0.0),
                size=(1.0, 0.1),
                velocity=(0, 0),
            ),
            *[
                Rectangle(
                    lower_left=(0.48, 0.9),
                    velocity=(0, -2.5),
                    size=(0.06, 0.06),
                    frame_threshold=i,
                )
                for i in range(1, 200)
            ],
        ],
    ),
    Configuration(
        name="Dam Break",
        geometries=[
            Rectangle(
                lower_left=(offset, offset),
                size=(0.3, 0.4),
                velocity=(0, 0),
            ),
        ],
    ),
    Configuration(
        name="Centered Dam Break",
        geometries=[
            Rectangle(
                lower_left=(0.35, offset),
                size=(0.3, 0.4),
                velocity=(0, 0),
            ),
        ],
    ),
    Configuration(
        name="Waterjet",
        geometries=[
            Rectangle(
                lower_left=(0.48, 0.9),
                velocity=(0, -2.5),
                size=(0.06, 0.06),
                frame_threshold=i,
            )
            for i in range(1, 200)
        ],
    ),
    Configuration(
        name="Spherefall",
        geometries=[
            Circle(
                center=(0.5, 0.5),
                velocity=(0, -1),
                radius=0.1,
            ),
        ],
    ),
    Configuration(
        name="Stationary Pool",
        geometries=[
            Rectangle(
                lower_left=(offset, offset),
                size=(1.0, 0.25),
                velocity=(0, 0),
            ),
        ],
    ),
]

# Sort alphabetically:
configuration_list.sort(key=lambda c: str.lower(c.name), reverse=False)
