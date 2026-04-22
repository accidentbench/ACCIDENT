from collections import Counter, defaultdict
from typing import Any, Dict, List

import numpy as np
import pygame

import carla
from typing import Tuple


class CarlaGUI:

    BOUNDING_BOX_COLOR = (248, 64, 24)
    COLLISION_COLOR = (180, 0, 255)
    TEXT_COLOR = (255, 255, 255)
    FONT_SIZE = 28
    BOUNDING_BOX_WIDTH = 5

    def __init__(
        self,
        display_size: tuple,
        segmentation_tags_mapping: Dict[str, str],
    ):
        """Pygame GUI to visualize created annotations onto the captured RGB image in real time.

        Args:
            display_size: Resolution.
            segmentation_tags_mapping: Mapping from tag number to cls name.
        """
        self.display_size = display_size
        self.segmentation_tags_mapping = segmentation_tags_mapping
        pygame.init()

        self.display = pygame.display.set_mode(
            self.display_size, pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        self.font = get_font(font_size=self.FONT_SIZE)
        self.clock = pygame.time.Clock()

    def quit(self) -> None:
        """Close the window."""
        pygame.quit()

    def run_draw(
        self,
        rgb_image: np.ndarray,
        segmented_image: np.ndarray,
        snapshot: carla.WorldSnapshot,
        tagged_segmented_bboxes: List[Dict[str, Any]] = None,
        lidar_points: np.ndarray = None,
        collision_bbox: tuple = None,
    ) -> None:
        """Draw a single frame with generated metadata.

        Args:
            rgb_image: Captured RGB image from a camera.
            segmented_image: Segmentation image of the same scene as the rbg camera.
            snapshot: Simulation snapshot with extra metadata.
            tagged_segmented_bboxes: List of retrieved bounding boxes with tags and contours.
            lidar_points: Lidar points projected to camera view.
            collision_bbox: Optional bounding box with the collision.

        """
        self.clock.tick()

        if rgb_image is not None:
            self.draw_image(rgb_image)
        if segmented_image is not None:
            self.draw_image(segmented_image, blend=True)
        if tagged_segmented_bboxes is not None:
            self.draw_2d_bboxes(
                [
                    (ann["tag"], ann["2d_bbox"], ann["contour"])
                    for ann in tagged_segmented_bboxes
                ],
                color=self.BOUNDING_BOX_COLOR,
            )
        if collision_bbox is not None:
            self.draw_2d_bboxes(
                [("collision", collision_bbox, None), ],
                color=self.COLLISION_COLOR,
            )
        if lidar_points is not None:
            self.draw_lidar_points(lidar_points)

        self.draw_info_panel(snapshot, tagged_segmented_bboxes)

        pygame.display.flip()

    def save_display(self, output_path: str) -> None:
        """Saves the windows as an image to a file."""
        pygame.image.save(self.display, output_path)

    def draw_info_panel(
        self, snapshot: carla.WorldSnapshot = None, bboxes: List[Dict[str, Any]] = None
    ) -> None:
        """Draw simulation information such as fps, and number of bboxes, etc. in the top left corner.

        Args:
            snapshot: World snapshot with metadata.
            bboxes: Plotted bounding boxes.

        """
        padding = 4
        line_height = self.font.get_linesize()
        tag_counts = count_unique_tags(bboxes)
        num_lines = 1 + bool(snapshot) + len(tag_counts) + 1

        # Background tile size
        tile_width = 310
        tile_height = padding * 2 + num_lines * (line_height + padding)

        # Transparent surface
        panel_surf = pygame.Surface((tile_width, tile_height), pygame.SRCALPHA)
        panel_surf.fill((100, 100, 100, 180))  # RGBA: gray with alpha

        # Blit tile first
        self.display.blit(panel_surf, (5, 5))

        # Draw text lines on top
        y = 5 + padding
        self.display.blit(
            self.font.render(f"{'FPS (real):':<13}{(self.clock.get_fps()):.2f}", True, self.TEXT_COLOR),
            (13, y)
        )
        y += line_height + padding
        if snapshot:
            fps = round(1.0 / snapshot.timestamp.delta_seconds)
            self.display.blit(
                self.font.render(f"{'FPS (sim):':<13}{fps}", True, self.TEXT_COLOR),
                (13, y)
            )
            y += line_height + padding
        if bboxes:
            for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True):
                tag = self.segmentation_tags_mapping[tag] + ":"
                self.display.blit(
                    self.font.render(f"{tag:<13}{count}", True, self.TEXT_COLOR),
                    (13, y)
                )
                y += line_height + padding

            self.display.blit(
                self.font.render(f"{'Total:':<13}{len(bboxes)}", True, self.TEXT_COLOR),
                (13, y)
            )


    def draw_image(self, image: np.ndarray, blend: bool = False) -> None:
        """Draw an image to the display.

        Args:
            image: [W,H,RGB] array.
            blend: Set transparency.

        """
        image_surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
        if blend:
            image_surface.set_alpha(100)
        self.display.blit(image_surface, (0, 0))

    def draw_2d_bboxes(self, bboxes: List[Dict[str, Any]], color: Tuple[int, int, int]) -> None:
        """Draw all 2d bboxes to the display.

        Args:
            bboxes: List of projected bounding boxes, tags, and contours.
            color: Bounding box color.

        """
        width, height = self.display.get_size()
        bb_surface = pygame.Surface((width, height))
        bb_surface.set_colorkey((0, 0, 0))
        for tag, bbox, _ in bboxes:
            # x_min, y_min
            # x_max, y_max
            self.draw_bbox(bb_surface, bbox, color)
            self.draw_tag_text_above_bbox(bb_surface, bbox, tag)
        self.display.blit(bb_surface, (0, 0))

    def draw_bbox(self, bb_surface: pygame.Surface, bbox: np.ndarray, color: Tuple[int, int, int]) -> None:
        """Draw a single bounding box onto the surface.

        Args:
            bb_surface: Surface to draw on.
            bbox: [[x_min, y_min], [x_max, y_max]]
            color: Bounding box color.

        """
        pygame.draw.line(
            bb_surface,
            color,
            bbox[0, :],
            (bbox[1, 0], bbox[0, 1]),
            width=self.BOUNDING_BOX_WIDTH,
        )
        pygame.draw.line(
            bb_surface,
            color,
            (bbox[1, 0], bbox[0, 1]),
            bbox[1, :],
            width=self.BOUNDING_BOX_WIDTH,
        )
        pygame.draw.line(
            bb_surface,
            color,
            bbox[1, :],
            (bbox[0, 0], bbox[1, 1]),
            width=self.BOUNDING_BOX_WIDTH,
        )
        pygame.draw.line(
            bb_surface,
            color,
            (bbox[0, 0], bbox[1, 1]),
            bbox[0, :],
            width=self.BOUNDING_BOX_WIDTH,
        )

    def draw_lidar_points(self, lidar_points: np.ndarray) -> None:
        """Draw all lidar points to the display."""
        width, height = self.display.get_size()
        surface = pygame.Surface((width, height))
        surface.set_colorkey((0, 0, 0))
        for x, y in lidar_points.tolist():
            pygame.draw.circle(surface, "pink", (x, y), 2)
        self.display.blit(surface, (0, 0))

    def draw_tag_text_above_bbox(
        self, surface: pygame.Surface, bbox: np.ndarray, tag: str
    ) -> None:
        """Draw class tag above the 2d bounding box."""
        tag = self.segmentation_tags_mapping.get(tag, tag)
        text = self.font.render(f"{tag}", True, self.TEXT_COLOR)
        text_width, text_height = self.font.size(f"{tag}")
        x_min, y_min = bbox[0, 0], bbox[0, 1]
        text_x = x_min
        text_y = y_min - text_height - 3

        text_surface = pygame.Surface(text.get_size())
        text_surface.fill((50, 50, 50))
        text_surface.blit(text, (0, 0))
        surface.blit(text_surface, (text_x, text_y))


def should_quit() -> bool:
    """Check in Escape button pressed. If so, return True."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def get_font(font_name: str = "dejavusansmono", font_size: int = 18) -> pygame.font.Font:
    """Get pygame font instance based on the font name and size."""
    fonts = [x for x in pygame.font.get_fonts()]
    font = font_name if font_name in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, font_size)


def count_unique_tags(tagged_segmented_bboxes: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count the number of tags per unique class. Return it as a `cls: count` dictionary."""
    tag_counts = defaultdict(int)
    for segmented_bbox in tagged_segmented_bboxes:
        tag = segmented_bbox["tag"]
        if tag is not None:
            tag_counts[tag] += 1
    return tag_counts





