from dataclasses import dataclass, field
from typing import Optional, List
from typing import List, Tuple
import numpy as np
import cv2
import supervision as sv


@dataclass
class BasketballCourtConfiguration:
    # NBA court dimensions in cm
    length: int = 2880  # 28.65m (94 feet) in cm
    width: int = 1530  # 15.24m (50 feet) in cm
    line_width: int = 1  # [cm] - width of all lines
    
    # Key (paint area) dimensions - NBA has a 16 feet wide key
    key_width: int = 488  # [cm] (16 feet)
    key_length: int = 580  # [cm] (19 feet)
    free_throw_line_distance: int = 580  # [cm] (19 feet) from baseline
    
    # Three-point line dimensions - NBA has 23.75ft (except corners at 22ft)
    three_point_line_radius: int = 724  # [cm] (23.75 feet)
    three_point_corner_distance: int = 669  # [cm] from sideline (creates 22ft in corners)
    three_point_line_straight_distance: int = 669  # [cm] (22 feet) from baseline
    
    # Backboard and rim - NBA standard
    backboard_width: int = 183  # [cm] (6 feet)
    backboard_distance: int = 122  # [cm] (4 feet) from baseline
    rim_diameter: int = 46  # [cm] (18 inches)
    rim_distance: int = 15  # [cm] (6 inches) from backboard

    @property
    def vertices(self) -> List[Tuple[int, int]]:
        half_width = self.width / 2
        half_length = self.length / 2
        
        # Calculate key points
        key_top = half_width + (self.key_width / 2)
        key_bottom = half_width - (self.key_width / 2)
        free_throw_line = self.free_throw_line_distance
        
        # Calculate three-point straight section points
        # These should be exactly at three_point_corner_distance from the center
        three_pt_straight_y_top = half_width + self.three_point_corner_distance
        three_pt_straight_y_bottom = half_width - self.three_point_corner_distance

        
        return [
            # Court outline
            (0, 0),  # 1 - corner
            (0, self.width),  # 2 - corner
            (self.length, self.width),  # 3 - corner
            (self.length, 0),  # 4 - corner
            
            # Center line
            (half_length, 0),  # 5
            (half_length, self.width),  # 6
            
           # Key (paint area) - left side
            (0, key_bottom),  # 7
            (free_throw_line, key_bottom),  # 8
            (free_throw_line, key_top),  # 9
            (0, key_top),  # 10
            
            # Key (paint area) - right side
            (self.length, key_bottom),  # 11
            (self.length - free_throw_line, key_bottom),  # 12
            (self.length - free_throw_line, key_top),  # 13
            (self.length, key_top),  # 14
        ]

    edges: List[Tuple[int, int]] = field(default_factory=lambda: [
        # Court outline (only sidelines and baselines)
        (1, 2), (2, 3), (3, 4), (4, 1),
        
        # Center line
        (5, 6),
        
        # Key (paint area) - left (only the outer lines)
        (7, 8), (8, 9), (9, 10),
        
        # Key (paint area) - right (only the outer lines)
        (11, 12), (12, 13), (13, 14),
    ])


def draw_court(
    config: BasketballCourtConfiguration,
    background_color: sv.Color = sv.Color(204, 153, 106),  # Dark green
    line_color: sv.Color = sv.Color.WHITE,
    padding: int = 10,
    line_thickness: int = 4,
    scale: float = 1
) -> np.ndarray:
    """
    Draws a simplified NBA basketball court with only essential lines.
    
    Args:
        config (BasketballCourtConfiguration): Configuration object.
        background_color (sv.Color): Court background color.
        line_color (sv.Color): Line color.
        padding (int): Padding around the court.
        line_thickness (int): Line thickness.
        scale (float): Scaling factor.
        
    Returns:
        np.ndarray: Image of the simplified basketball court.
    """
    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)
    
    court_image = np.ones(
        (scaled_width + 2 * padding,
         scaled_length + 2 * padding, 3),
        dtype=np.uint8
    ) * np.array(background_color.as_bgr(), dtype=np.uint8)

    # Draw edges
    for start, end in config.edges:
        point1 = (int(config.vertices[start - 1][0] * scale) + padding,
                  int(config.vertices[start - 1][1] * scale) + padding)
        point2 = (int(config.vertices[end - 1][0] * scale) + padding,
                  int(config.vertices[end - 1][1] * scale) + padding)
        cv2.line(
            img=court_image,
            pt1=point1,
            pt2=point2,
            color=line_color.as_bgr(),
            thickness=line_thickness
        )

    return court_image


def draw_points_on_pitch(
    config: BasketballCourtConfiguration,
    xy: np.ndarray,
    face_color: sv.Color = sv.Color.RED,
    edge_color: sv.Color = sv.Color.BLACK,
    radius: int = 10,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Draws points on a soccer pitch.

    Args:
        config (SoccerPitchConfiguration): Configuration object containing the
            dimensions and layout of the pitch.
        xy (np.ndarray): Array of points to be drawn, with each point represented by
            its (x, y) coordinates.
        face_color (sv.Color, optional): Color of the point faces.
            Defaults to sv.Color.RED.
        edge_color (sv.Color, optional): Color of the point edges.
            Defaults to sv.Color.BLACK.
        radius (int, optional): Radius of the points in pixels.
            Defaults to 10.
        thickness (int, optional): Thickness of the point edges in pixels.
            Defaults to 2.
        padding (int, optional): Padding around the pitch in pixels.
            Defaults to 50.
        scale (float, optional): Scaling factor for the pitch dimensions.
            Defaults to 0.1.
        pitch (Optional[np.ndarray], optional): Existing pitch image to draw points on.
            If None, a new pitch will be created. Defaults to None.

    Returns:
        np.ndarray: Image of the soccer pitch with points drawn on it.
    """
    if pitch is None:
        pitch = draw_court(
            config=config,
            padding=padding,
            scale=scale
        )

    for point in xy:
        scaled_point = (
            int(point[0] * scale) + padding,
            int(point[1] * scale) + padding
        )
        cv2.circle(
            img=pitch,
            center=scaled_point,
            radius=radius,
            color=face_color.as_bgr(),
            thickness=-1
        )
        cv2.circle(
            img=pitch,
            center=scaled_point,
            radius=radius,
            color=edge_color.as_bgr(),
            thickness=thickness
        )

    return pitch


def draw_paths_on_pitch(
    config: BasketballCourtConfiguration,
    paths: List[np.ndarray],
    color: sv.Color = sv.Color.WHITE,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Draws paths on a soccer pitch.

    Args:
        config (SoccerPitchConfiguration): Configuration object containing the
            dimensions and layout of the pitch.
        paths (List[np.ndarray]): List of paths, where each path is an array of (x, y)
            coordinates.
        color (sv.Color, optional): Color of the paths.
            Defaults to sv.Color.WHITE.
        thickness (int, optional): Thickness of the paths in pixels.
            Defaults to 2.
        padding (int, optional): Padding around the pitch in pixels.
            Defaults to 50.
        scale (float, optional): Scaling factor for the pitch dimensions.
            Defaults to 0.1.
        pitch (Optional[np.ndarray], optional): Existing pitch image to draw paths on.
            If None, a new pitch will be created. Defaults to None.

    Returns:
        np.ndarray: Image of the soccer pitch with paths drawn on it.
    """
    if pitch is None:
        pitch = draw_court(
            config=config,
            padding=padding,
            scale=scale
        )

    for path in paths:
        scaled_path = [
            (
                int(point[0] * scale) + padding,
                int(point[1] * scale) + padding
            )
            for point in path if point.size > 0
        ]

        if len(scaled_path) < 2:
            continue

        for i in range(len(scaled_path) - 1):
            cv2.line(
                img=pitch,
                pt1=scaled_path[i],
                pt2=scaled_path[i + 1],
                color=color.as_bgr(),
                thickness=thickness
            )

        return pitch


def draw_pitch_voronoi_diagram(
    config: BasketballCourtConfiguration,
    team_1_xy: np.ndarray,
    team_2_xy: np.ndarray,
    team_1_color: sv.Color = sv.Color.RED,
    team_2_color: sv.Color = sv.Color.WHITE,
    opacity: float = 0.5,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Draws a Voronoi diagram on a soccer pitch representing the control areas of two
    teams.

    Args:
        config (SoccerPitchConfiguration): Configuration object containing the
            dimensions and layout of the pitch.
        team_1_xy (np.ndarray): Array of (x, y) coordinates representing the positions
            of players in team 1.
        team_2_xy (np.ndarray): Array of (x, y) coordinates representing the positions
            of players in team 2.
        team_1_color (sv.Color, optional): Color representing the control area of
            team 1. Defaults to sv.Color.RED.
        team_2_color (sv.Color, optional): Color representing the control area of
            team 2. Defaults to sv.Color.WHITE.
        opacity (float, optional): Opacity of the Voronoi diagram overlay.
            Defaults to 0.5.
        padding (int, optional): Padding around the pitch in pixels.
            Defaults to 50.
        scale (float, optional): Scaling factor for the pitch dimensions.
            Defaults to 0.1.
        pitch (Optional[np.ndarray], optional): Existing pitch image to draw the
            Voronoi diagram on. If None, a new pitch will be created. Defaults to None.

    Returns:
        np.ndarray: Image of the soccer pitch with the Voronoi diagram overlay.
    """
    if pitch is None:
        pitch = draw_court(
            config=config,
            padding=padding,
            scale=scale
        )

    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)

    voronoi = np.zeros_like(pitch, dtype=np.uint8)

    team_1_color_bgr = np.array(team_1_color.as_bgr(), dtype=np.uint8)
    team_2_color_bgr = np.array(team_2_color.as_bgr(), dtype=np.uint8)

    y_coordinates, x_coordinates = np.indices((
        scaled_width + 2 * padding,
        scaled_length + 2 * padding
    ))

    y_coordinates -= padding
    x_coordinates -= padding

    def calculate_distances(xy, x_coordinates, y_coordinates):
        return np.sqrt((xy[:, 0][:, None, None] * scale - x_coordinates) ** 2 +
                       (xy[:, 1][:, None, None] * scale - y_coordinates) ** 2)

    distances_team_1 = calculate_distances(team_1_xy, x_coordinates, y_coordinates)
    distances_team_2 = calculate_distances(team_2_xy, x_coordinates, y_coordinates)

    min_distances_team_1 = np.min(distances_team_1, axis=0)
    min_distances_team_2 = np.min(distances_team_2, axis=0)

    control_mask = min_distances_team_1 < min_distances_team_2

    voronoi[control_mask] = team_1_color_bgr
    voronoi[~control_mask] = team_2_color_bgr

    overlay = cv2.addWeighted(voronoi, opacity, pitch, 1 - opacity, 0)

    return overlay



