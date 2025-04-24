from src.configurations import Circle, Rectangle, Configuration

# Width of the bounding box:
# TODO: scale and translate positions to coordinates of bounding box
offset = 0.0234375

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
                    lower_left=(0.48, 0.48),
                    frame_threshold=i,
                )
                for i in range(1, 300)
            ],
        ],
        E=5e5,  # Young's modulus (1.4e5)
        nu=0.45,  # Poisson's ratio (0.2)
        zeta=10,  # Hardening coefficient (10)
        theta_c=2.5e-2,  # Critical compression (2.5e-2)
        theta_s=5.0e-3,  # Critical stretch (7.5e-3)
        ambient_temperature=20.0,
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
        E=5e5,  # Young's modulus (1.4e5)
        nu=0.45,  # Poisson's ratio (0.2)
        zeta=10,  # Hardening coefficient (10)
        theta_c=2.5e-2,  # Critical compression (2.5e-2)
        theta_s=5.0e-3,  # Critical stretch (7.5e-3)
        ambient_temperature=20.0,
    ),
    Configuration(
        name="Simple Spout Source [Water]",
        geometries=[
            Rectangle(
                lower_left=(0.48, 0.48),
                velocity=(0, -3),
                size=(0.04, 0.04),
                frame_threshold=i,
            )
            for i in range(1, 300)
        ],
        E=5.5e5,  # Young's modulus (1.4e5)
        nu=0.45,  # Poisson's ratio (0.2)
        zeta=1,  # Hardening coefficient (10)
        theta_c=2.5e-2,  # Critical compression (2.5e-2)
        theta_s=5.0e-3,  # Critical stretch (7.5e-3)
        ambient_temperature=20.0,
    ),
    Configuration(
        name="Spherefall [Water]",
        geometries=[Circle(center=(0.5, 0.35), velocity=(0, -2), radius=0.08)],
        E=1.4e5,  # Young's modulus (1.4e5)
        nu=0.2,  # Poisson's ratio (0.2)
        zeta=10,  # Hardening coefficient (10)
        theta_c=2.5e-2,  # Critical compression (2.5e-2)
        theta_s=5.0e-3,  # Critical stretch (7.5e-3)
        ambient_temperature=-20.0,
    ),
]

# Sort by length in descending order:
configuration_list.sort(key=lambda c: len(c.name), reverse=True)
