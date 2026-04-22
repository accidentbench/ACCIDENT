import os
import time

import open3d as o3d


def set_viewpoint(vis):
    ctr = vis.get_view_control()
    # Define a consistent viewpoint
    ctr.set_front([0, 0, -1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, -1, 0])
    ctr.set_zoom(0.1)
    # ctr.translate(-300, 200)


def main():
    # Directory containing PLY files
    ply_dir = "../runs/out/lidar"

    # Get a list of all PLY files in the directory
    ply_files = [f for f in os.listdir(ply_dir) if f.endswith(".ply")]

    # Sort the files for sequential display
    ply_files.sort()

    # Create a visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().background_color = [0.05, 0.05, 0.05]
    vis.get_render_option().point_size = 4
    vis.get_render_option().show_coordinate_frame = True
    # add_open3d_axis(vis)

    for i, ply_file in enumerate(ply_files):
        # Load the PLY file
        pcd = o3d.io.read_point_cloud(os.path.join(ply_dir, ply_file))

        # Clear previous geometry and add the new one
        vis.clear_geometries()
        vis.add_geometry(pcd)

        set_viewpoint(vis)

        # Update the visualizer and wait for a bit
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(f"../video/{i:05d}.png")
        time.sleep(0.05)  # Display each PLY for 2 seconds

    # Close the visualizer
    vis.destroy_window()


if __name__ == "__main__":
    main()
