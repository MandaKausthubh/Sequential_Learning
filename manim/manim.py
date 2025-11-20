from manim import *
import numpy as np

class LossSurfaceTopView(ThreeDScene):
    def construct(self):

        # More top-down camera while showing curvature
        self.set_camera_orientation(phi=75 * DEGREES, theta=-30 * DEGREES)

        # Axes
        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[0, 6, 1],
            x_length=6,
            y_length=6,
            z_length=4,
        )
        self.add(axes)

        def loss(u, v):
            return 0.2 * (u ** 2 + 0.5 * v ** 2)

        # Surface
        surface = Surface(
            lambda u, v: axes.c2p(u, v, loss(u, v)),
            u_range=[-3, 3],
            v_range=[-3, 3],
            resolution=(32, 32),
        )

        # Add clear fill
        surface.set_fill_by_value(
            axes=axes,
            colors=[(YELLOW, 0), (ORANGE, 2), (RED, 5)],
        )
        surface.set_opacity(0.85)
        surface.set_stroke(width=0.25)

        self.add(surface)

        # Base point
        base = axes.c2p(0, 0, loss(0, 0))

        # Δ (original update)
        delta_end = axes.c2p(0.6, 1.0, loss(0.6, 1.0))
        delta_arrow = Arrow3D(base, delta_end, color=BLACK, stroke_width=6)
        delta_label = Text("Δ", font_size=40, color=BLACK)
        delta_label.move_to(delta_arrow.get_end()).shift(UP * 0.4 + OUT * 0.7)

        self.add(delta_arrow, delta_label)

        # Q_k direction (high curvature)
        q_end = axes.c2p(0, 1.7, loss(0, 1.7))
        q_arrow = Arrow3D(base, q_end, color=GRAY)
        q_label = Text("Qₖ", font_size=36, color=GRAY)
        q_label.move_to(q_arrow.get_end()).shift(UP * 0.3 + OUT * 0.5)

        self.add(q_arrow, q_label)

        # Δ' (projection)
        proj_end = axes.c2p(1.3, 0.1, loss(1.3, 0.1))
        proj_arrow = Arrow3D(base, proj_end, color=BLUE, stroke_width=8)
        proj_label = Text("Δ′ = (I − QQᵀ) Δ", font_size=36, color=BLUE)
        proj_label.move_to(proj_arrow.get_end()).shift(DOWN * 0.4 + OUT * 0.8)

        self.add(proj_arrow, proj_label)

        # Null space label (lift upward for visibility)
        null_label = Text("Null-space / flat valley", font_size=32, color=YELLOW)
        null_label.move_to(axes.c2p(1.5, -1.5, 0.3)).shift(OUT * 1.2)
        self.add(null_label)

        # High curvature region label
        steep_label = Text("High curvature region", font_size=28, color=RED)
        steep_label.move_to(axes.c2p(-1.5, 2, 2.8)).shift(OUT * 0.8)
        self.add(steep_label)

        self.wait(3)
