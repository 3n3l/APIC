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
                    size=(0.04, 0.04),
                    velocity=(0, -2),
                    lower_left=(0.48, 0.8),
                    frame_threshold=i,
                )
                for i in range(1, 300)
            ],
        ],
    ),
    Configuration(
        name="Dam Break",
        geometries=[
            Rectangle(
                lower_left=(offset, offset),
                size=(0.5 - offset, 0.5 - offset),
                velocity=(0, 0),
            ),
        ],
    ),
    Configuration(
        name="Waterjet",
        geometries=[
            Rectangle(
                lower_left=(0.48, 0.8),
                velocity=(0, -2),
                size=(0.04, 0.04),
                frame_threshold=i,
            )
            for i in range(1, 300)
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
