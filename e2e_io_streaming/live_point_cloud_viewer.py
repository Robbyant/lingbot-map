import numpy as np
import matplotlib.cm as cm
import viser
import viser.transforms as tf

from lingbot_map.vis.point_cloud_viewer import PointCloudViewer


class LivePointCloudViewer(PointCloudViewer):
    def __init__(
        self,
        port: int = 8080,
        show_camera: bool = True,
        vis_threshold: float = 1.0,
        size: int = 512,
        downsample_factor: int = 10,
        point_size: float = 0.00001,
        use_point_map: bool = False,
        mask_sky: bool = False,
        image_folder: str | None = None,
        sky_mask_dir: str | None = None,
        sky_mask_visualization_dir: str | None = None,
    ):
        self.model = None
        self.size = size
        self.state_args = None
        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
        self.device = "cpu"
        self.conf_list = None
        self.vis_threshold = vis_threshold
        self.point_size = point_size
        self.show_camera = show_camera
        self.use_point_map = use_point_map
        self.mask_sky = mask_sky
        self.image_folder = image_folder
        self.sky_mask_dir = sky_mask_dir
        self.sky_mask_visualization_dir = sky_mask_visualization_dir

        self.original_images = []
        self.orig_img_list = []
        self.image_mask = None
        self.on_replay = False
        self.vis_pts_list = []
        self.traj_list = []
        self.via_points = []
        self.pcs = {}
        self.all_steps = []
        self.cam_dict = {"focal": [], "pp": [], "R": [], "t": []}
        self.num_frames = 0
        self.camera_colors = cm.get_cmap("viridis")(np.array([0.0]))

        self._setup_gui()
        self.server.on_client_connect(self._connect_client)
        self.server.scene.add_frame("/frames", show_axes=False)
        self.frame_nodes = []
        self._setup_live_gui(downsample_factor)

    def _setup_live_gui(self, downsample_factor: int) -> None:
        self.downsample_slider.value = downsample_factor
        self.gui_timestep = self.server.gui.add_slider(
            "Current Frame",
            min=0,
            max=0,
            step=1,
            initial_value=0,
            disabled=False,
        )
        self.follow_latest_checkbox = self.server.gui.add_checkbox(
            "Follow Latest Frame",
            initial_value=True,
        )

        @self.gui_timestep.on_update
        def _(_) -> None:
            self._refresh_live_visibility()

    def _refresh_live_visibility(self) -> None:
        if not self.frame_nodes:
            return

        current_timestep = int(self.gui_timestep.value)
        current_timestep = max(0, min(current_timestep, len(self.frame_nodes) - 1))
        if self.current_frame_image is not None and current_timestep < len(self.original_images):
            self.current_frame_image.image = self.original_images[current_timestep]

        with self.server.atomic():
            for i, frame_node in enumerate(self.frame_nodes):
                frame_node.visible = (i <= current_timestep) if not self.fourd else (i == current_timestep)
        self.server.flush()

    def _refresh_camera_gradient(self) -> None:
        num_cameras = max(len(self.all_steps), 1)
        if num_cameras > 1:
            normalized = np.array(list(range(num_cameras))) / (num_cameras - 1)
        else:
            normalized = np.array([0.0])
        self.camera_colors = cm.get_cmap("viridis")(normalized)

    def append_prediction(self, pred_dict: dict) -> dict:
        existing_images = list(self.original_images)
        pc_list, color_list, conf_list, cam_dict = self._process_pred_dict(
            pred_dict=pred_dict,
            use_point_map=self.use_point_map,
            mask_sky=self.mask_sky,
            image_folder=self.image_folder,
            sky_mask_dir=self.sky_mask_dir,
            sky_mask_visualization_dir=self.sky_mask_visualization_dir,
        )
        new_images = list(self.original_images)
        self.original_images = existing_images + new_images

        step = len(self.all_steps)
        self.pcs[step] = {
            "pc": pc_list[0],
            "color": color_list[0],
            "conf": conf_list[0],
            "edge_color": None,
        }
        self.all_steps.append(step)
        self.num_frames = len(self.all_steps)

        self.cam_dict["focal"].append(cam_dict["focal"][0])
        self.cam_dict["pp"].append(cam_dict["pp"][0])
        self.cam_dict["R"].append(cam_dict["R"][0])
        self.cam_dict["t"].append(cam_dict["t"][0])
        self._refresh_camera_gradient()

        frame_node = self.server.scene.add_frame(f"/frames/{step}", show_axes=False)
        self.frame_nodes.append(frame_node)
        self.add_pc(step)
        if self.show_camera:
            downsample = max(1, int(self.camera_downsample_slider.value))
            if step % downsample == 0:
                self.add_camera(step)

        self.gui_timestep.max = max(0, self.num_frames - 1)
        if self.follow_latest_checkbox.value or self.num_frames == 1:
            self.gui_timestep.value = self.num_frames - 1
        self._refresh_live_visibility()

        R = self.cam_dict["R"][step]
        t = self.cam_dict["t"][step]
        return {
            "points_added": int(len(self.vis_pts_list[-1])) if self.vis_pts_list else 0,
            "total_points": int(sum(len(x) for x in self.vis_pts_list)),
            "pose": {
                "position": np.asarray(t, dtype=np.float32).tolist(),
                "quaternion_wxyz": tf.SO3.from_matrix(R).wxyz.astype(np.float32).tolist(),
            },
        }
