import ast
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from data_generation.robot_bev.sources import habitat_common
from data_generation.robot_bev.sources.replica import (
    BEV_LABEL_SOURCE,
    MAP_CLASSES,
    REPLICA_SEMANTIC_ORIENT_FRONT,
    REPLICA_SEMANTIC_ORIENT_UP,
    configure_replica_semantic_orientation,
    generation_parameters,
    make_bev_labels,
    make_parser,
    run_generation,
    semantic_category_to_map_class,
)


class FakeState:
    def __init__(self, pose):
        self.pose = np.asarray(pose, dtype=np.float32)
        self.position = self.pose[:3, 3].copy()


class FakeAgent:
    def __init__(self, initial_state, transitions):
        self.state = initial_state
        self.transitions = list(transitions)
        self.actions = []

    def act(self, action):
        self.actions.append(action)
        state, collided = self.transitions.pop(0)
        self.state = state
        return collided

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state


class FakeSimulator:
    def __init__(self, initial_state, transitions):
        self.agent = FakeAgent(initial_state, transitions)
        self.sensor_calls = 0
        self.step_calls = 0
        self.world_steps = []

    def get_agent(self, agent_id):
        assert agent_id == 0
        return self.agent

    def get_sensor_observations(self):
        self.sensor_calls += 1
        return {"frame": self.sensor_calls}

    def step(self, action):
        self.step_calls += 1
        collided = self.agent.act(action)
        self.step_world(1.0 / 60.0)
        observations = self.get_sensor_observations()
        observations["collided"] = collided
        return observations

    def step_world(self, dt):
        self.world_steps.append(dt)


def make_pose(x=0.0, y=0.0, z=0.0):
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = [x, y, z]
    return pose


def make_transition_args(enable_stair_filter=False):
    return SimpleNamespace(
        scene="office_1",
        seed=7,
        timestamp_start=1_000_000,
        timestamp_step=100_000,
        enable_stair_filter=enable_stair_filter,
        stair_check_radius=0.5,
        max_floor_height_delta=0.03,
    )


def canonical_record(frame_id, pose):
    return {
        "frame_id": frame_id,
        "timestamp": 1_000_000 + frame_id * 100_000,
        "T_map_base": np.asarray(pose).tolist(),
    }


def test_replica_mapping_uses_canonical_classes():
    assert MAP_CLASSES == (
        "floor",
        "carpet",
        "wall",
        "furniture",
        "door",
        "clutter",
    )
    assert semantic_category_to_map_class("table") == "furniture"
    assert semantic_category_to_map_class("door") == "door"
    assert semantic_category_to_map_class("wall") == "wall"
    assert semantic_category_to_map_class("rug") == "carpet"
    assert semantic_category_to_map_class("ceiling") is None
    assert semantic_category_to_map_class("ceiling light") is None
    assert semantic_category_to_map_class("chandelier") is None
    assert semantic_category_to_map_class("lamp") == "clutter"


def test_bev_labels_use_semantic_projection_for_floor():
    xbound = (0.0, 1.0, 0.5)
    ybound = (-0.5, 0.5, 0.5)
    valid_mask = np.ones((2, 2), dtype=np.uint8)
    points = np.array(
        [
            [0.25, 0.0, 0.0, 0.0, 0.0],
            [0.75, -0.25, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    semantic_ids = np.array([10, 20], dtype=np.int64)

    labels = make_bev_labels(
        [(points, semantic_ids, np.zeros(3, dtype=np.float32))],
        {10: "floor", 20: "furniture"},
        xbound,
        ybound,
        (-0.5, 2.0),
        valid_mask,
    )

    assert labels[MAP_CLASSES.index("floor"), 0, 1] == 1
    assert labels[MAP_CLASSES.index("furniture"), 1, 0] == 1


def test_bev_labels_filter_semantic_points_by_zbound():
    xbound = (0.0, 1.0, 0.5)
    ybound = (-0.5, 0.5, 0.5)
    zbound = (-0.5, 2.0)
    valid_mask = np.ones((2, 2), dtype=np.uint8)
    points = np.array(
        [
            [0.25, 0.0, 0.0, 0.0, 0.0],
            [0.25, 0.0, 2.1, 0.0, 0.0],
            [0.25, 0.0, -0.5, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    semantic_ids = np.array([10, 20, 30], dtype=np.int64)

    labels = make_bev_labels(
        [(points, semantic_ids, np.zeros(3, dtype=np.float32))],
        {10: "floor", 20: "wall", 30: "door"},
        xbound,
        ybound,
        zbound,
        valid_mask,
    )

    assert labels[MAP_CLASSES.index("floor"), 0, 1] == 1
    assert labels[MAP_CLASSES.index("wall")].sum() == 0
    assert labels[MAP_CLASSES.index("door")].sum() == 0


def test_schema_writer_and_validator_do_not_import_habitat():
    for relative in ("schema.py", "writer.py", "validator.py", "geometry_checks.py"):
        path = Path("data_generation/robot_bev") / relative
        tree = ast.parse(path.read_text())
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported.update(
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        )
        assert "habitat_sim" not in imported


def test_cli_contract_requires_dataset_id_and_removes_generation_sweeps():
    parser = make_parser()
    help_text = parser.format_help()

    for option in ("--dataset-id", "--split-file", "--resume", "--preflight-only"):
        assert option in help_text
    assert "--num-sweeps" not in help_text

    with pytest.raises(SystemExit):
        parser.parse_args(["--dataset", "replica.scene_dataset_config.json"])


def test_generation_parameters_capture_artifact_contract(
    tmp_path, monkeypatch
):
    split_file = tmp_path / "splits.json"
    split_contents = {
        "train": ["office_0"],
        "val": ["office_1"],
        "test": [],
    }
    split_file.write_text(json.dumps(split_contents), encoding="utf-8")
    args = make_parser().parse_args(
        [
            "--dataset",
            "replica.scene_dataset_config.json",
            "--dataset-id",
            "replica_robot_bev_v4",
            "--scenes",
            "office_0",
            "office_1",
            "--split-file",
            str(split_file),
            "--output-dir",
            str(tmp_path / "output"),
            "--resume",
            "--preflight-only",
            "--gpu-id",
            "2",
            "--save-visualization",
        ]
    )
    monkeypatch.setattr(
        habitat_common,
        "habitat_sim",
        SimpleNamespace(__version__="0.2.2"),
    )

    parameters = generation_parameters(args)

    assert parameters["dataset"] == "replica.scene_dataset_config.json"
    assert parameters["requested_scenes"] == ["office_0", "office_1"]
    assert parameters["split_file_contents"] == split_contents
    assert parameters["habitat_sim_version"] == "0.2.2"
    assert parameters["width"] == 640
    assert parameters["xbound"] == [0.0, 3.0, 0.02]
    assert parameters["zbound"] == [-0.5, 2.0]
    assert parameters["semantic_sensor"] is True
    assert parameters["semantic_orient_up"] == [0.0, 1.0, 0.0]
    assert parameters["semantic_orient_front"] == [0.0, 0.0, -1.0]
    assert "table" in parameters["semantic_category_groups"]["furniture"]
    assert parameters["semantic_category_groups"]["door"] == ["door"]
    assert parameters["fallback_semantic_class"] == "clutter"
    assert parameters["bev_label_source"] == BEV_LABEL_SOURCE
    assert "unknown" in parameters["ignored_semantic_categories"]
    assert "ceiling light" in parameters["ignored_semantic_categories"]
    for excluded in (
        "output_dir",
        "resume",
        "preflight_only",
        "gpu_id",
        "save_visualization",
        "save_ply",
    ):
        assert excluded not in parameters


def test_replica_semantic_orientation_overrides_stage_defaults(
    tmp_path, monkeypatch
):
    stage_path = (tmp_path / "replica_stage.stage_config.json").resolve()
    template = SimpleNamespace(
        semantic_orient_up=None,
        semantic_orient_front=None,
    )

    class FakeStageManager:
        def __init__(self):
            self.registration = None

        def get_template_by_handle(self, handle):
            assert handle == str(stage_path)
            return template

        def register_template(
            self, value, handle, force_registration=False
        ):
            self.registration = (value, handle, force_registration)
            return 3

    manager = FakeStageManager()
    mediator = SimpleNamespace(stage_template_manager=manager)
    fake_habitat = SimpleNamespace(
        metadata=SimpleNamespace(
            MetadataMediator=lambda unused_config: mediator
        )
    )
    monkeypatch.setattr(
        habitat_common, "require_habitat_sim", lambda: fake_habitat
    )
    monkeypatch.setattr(
        habitat_common,
        "mn",
        SimpleNamespace(Vector3=lambda values: tuple(values)),
    )
    cfg = SimpleNamespace(sim_cfg=object(), metadata_mediator=None)

    configure_replica_semantic_orientation(cfg, stage_path)

    assert cfg.metadata_mediator is mediator
    assert template.semantic_orient_up == REPLICA_SEMANTIC_ORIENT_UP
    assert template.semantic_orient_front == REPLICA_SEMANTIC_ORIENT_FRONT
    assert manager.registration == (template, str(stage_path), True)


def test_far_depth_rays_still_cover_canonical_bev():
    depth = np.array([[6.0]], dtype=np.float32)
    intrinsic = np.eye(3, dtype=np.float32)
    t_base_camera_habitat = np.array(
        [
            [0.0, 0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    points, _ = habitat_common.depth_to_points(
        depth,
        intrinsic,
        t_base_camera_habitat,
        max_depth=4.0,
        stride=1,
        max_points=10,
    )
    observation_points, _ = habitat_common.depth_to_points(
        depth,
        intrinsic,
        t_base_camera_habitat,
        max_depth=float("inf"),
        stride=1,
        max_points=10,
    )

    assert points.shape == (0, 5)
    np.testing.assert_allclose(observation_points[0, :3], [6.0, 0.0, 0.0])

    valid_mask = habitat_common.make_observation_mask(
        [(observation_points, None, np.zeros(3, dtype=np.float32))],
        (0.0, 3.0, 0.02),
        (-1.5, 1.5, 0.02),
    )
    assert valid_mask[:, 75].all()


def test_run_generation_uses_one_root_writer_for_all_scenes(
    tmp_path, monkeypatch
):
    split_file = tmp_path / "splits.json"
    split_file.write_text(
        json.dumps(
            {
                "train": ["office_0"],
                "val": ["office_1"],
                "test": [],
            }
        ),
        encoding="utf-8",
    )
    args = make_parser().parse_args(
        [
            "--dataset",
            "replica.scene_dataset_config.json",
            "--dataset-id",
            "replica_robot_bev_v4",
            "--scenes",
            "office_0",
            "office_1",
            "--split-file",
            str(split_file),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )
    monkeypatch.setattr(
        habitat_common,
        "require_habitat_sim",
        lambda: SimpleNamespace(__version__="0.2.2"),
    )

    from data_generation.robot_bev.sources import replica

    monkeypatch.setattr(
        replica,
        "validate_replica_scene",
        lambda dataset, scene: SimpleNamespace(ptex_atlas_count=3),
    )
    writer_calls = []
    generated = []

    class FakeWriter:
        def __init__(self, **kwargs):
            writer_calls.append(kwargs)
            self.finalized = False

        def finalize_dataset(self):
            self.finalized = True
            return {"status": "complete"}

    monkeypatch.setattr(replica, "RobotBEVWriter", FakeWriter)
    monkeypatch.setattr(
        replica,
        "generate_scene",
        lambda scene_args, scene_files, split, writer: generated.append(
            (scene_args.scene, split, scene_args.seed, scene_args.timestamp_start, writer)
        ),
    )

    run_generation(args)

    assert len(writer_calls) == 1
    assert writer_calls[0]["root"] == Path(args.output_dir)
    assert writer_calls[0]["splits"] == {
        "train": ["office_0"],
        "val": ["office_1"],
        "test": [],
    }
    assert writer_calls[0]["source_dataset"] == "replica_v1"
    assert [item[:2] for item in generated] == [
        ("office_0", "train"),
        ("office_1", "val"),
    ]
    assert generated[1][2] == args.seed + 1
    assert generated[1][3] == args.timestamp_start + args.scene_timestamp_stride
    assert generated[0][4].finalized is True


def test_resume_replay_retains_collision_state_for_next_frame(monkeypatch):
    from data_generation.robot_bev.sources import replica

    initial = FakeState(make_pose())
    collided = FakeState(make_pose())
    next_state = FakeState(make_pose(x=0.1))
    sim = FakeSimulator(
        initial,
        [(collided, True), (next_state, False)],
    )
    args = make_transition_args()
    monkeypatch.setattr(
        replica, "map_from_base_matrix", lambda state: state.pose
    )
    monkeypatch.setattr(
        replica,
        "initialize_agent",
        lambda target, unused_args: target.agent.set_state(initial),
    )
    records = [
        canonical_record(0, initial.pose),
        canonical_record(1, collided.pose),
    ]

    trajectory = replica._initialize_trajectory(sim, args, records)

    assert trajectory.last_collided is True
    assert trajectory.stair_recoveries == 0
    assert sim.sensor_calls == 0
    assert sim.step_calls == 0
    expected_action = replica.next_action(
        True,
        random.Random((args.seed + 1) * 1_000_003 + 2),
    )

    result = replica._advance_trajectory(
        sim,
        args,
        frame_idx=2,
        trajectory=trajectory,
        render_observations=True,
    )

    assert sim.agent.actions[-1] == expected_action
    assert result.trajectory.last_collided is False
    assert result.observations["frame"] == 1


def test_resume_replay_retains_stair_recovery_state(monkeypatch):
    from data_generation.robot_bev.sources import replica

    initial = FakeState(make_pose())
    moved = FakeState(make_pose(x=0.05))
    sim = FakeSimulator(initial, [(moved, False)])
    args = make_transition_args(enable_stair_filter=True)
    monkeypatch.setattr(
        replica, "map_from_base_matrix", lambda state: state.pose
    )
    monkeypatch.setattr(
        replica,
        "initialize_agent",
        lambda target, unused_args: target.agent.set_state(initial),
    )
    monkeypatch.setattr(replica, "is_floor_level_safe", lambda *args: False)

    turns = [-np.pi / 2.0, np.pi / 2.0, np.pi]

    def fake_turn_agent_away(target, previous_state, rng):
        recovered_pose = previous_state.pose.copy()
        recovered_pose[1, 3] = rng.choice(turns)
        target.agent.set_state(FakeState(recovered_pose))

    monkeypatch.setattr(replica, "turn_agent_away", fake_turn_agent_away)
    expected_rng = random.Random((args.seed + 1) * 1_000_003 + 1)
    replica.next_action(False, expected_rng)
    expected_pose = initial.pose.copy()
    expected_pose[1, 3] = expected_rng.choice(turns)
    records = [
        canonical_record(0, initial.pose),
        canonical_record(1, expected_pose),
    ]

    trajectory = replica._initialize_trajectory(sim, args, records)

    assert trajectory.last_collided is True
    assert trajectory.last_stair_recovery is True
    assert trajectory.stair_recoveries == 1
    np.testing.assert_array_equal(sim.agent.state.pose, expected_pose)
    assert sim.sensor_calls == 0
    assert sim.step_calls == 0


def test_resume_replay_rejects_pose_divergence_with_context(monkeypatch):
    from data_generation.robot_bev.sources import replica

    initial = FakeState(make_pose())
    replayed = FakeState(make_pose(x=0.05))
    sim = FakeSimulator(initial, [(replayed, False)])
    args = make_transition_args()
    monkeypatch.setattr(
        replica, "map_from_base_matrix", lambda state: state.pose
    )
    monkeypatch.setattr(
        replica,
        "initialize_agent",
        lambda target, unused_args: target.agent.set_state(initial),
    )
    records = [
        canonical_record(0, initial.pose),
        canonical_record(1, make_pose(x=0.25)),
    ]

    with pytest.raises(
        RuntimeError,
        match=r"office_1.*frame 000001.*atol=.*max_abs_error",
    ):
        replica._initialize_trajectory(sim, args, records)


def test_resume_replay_compares_frame_zero_pose(monkeypatch):
    from data_generation.robot_bev.sources import replica

    initial = FakeState(make_pose())
    sim = FakeSimulator(initial, [])
    args = make_transition_args()
    monkeypatch.setattr(
        replica, "map_from_base_matrix", lambda state: state.pose
    )
    monkeypatch.setattr(
        replica,
        "initialize_agent",
        lambda target, unused_args: target.agent.set_state(initial),
    )

    with pytest.raises(RuntimeError, match=r"frame 000000.*max_abs_error"):
        replica._initialize_trajectory(
            sim,
            args,
            [canonical_record(0, make_pose(y=0.1))],
        )


def test_resume_replay_rejects_manifest_timestamp_divergence(monkeypatch):
    from data_generation.robot_bev.sources import replica

    initial = FakeState(make_pose())
    sim = FakeSimulator(initial, [])
    args = make_transition_args()
    monkeypatch.setattr(
        replica, "map_from_base_matrix", lambda state: state.pose
    )
    monkeypatch.setattr(
        replica,
        "initialize_agent",
        lambda target, unused_args: target.agent.set_state(initial),
    )
    record = canonical_record(0, initial.pose)
    record["timestamp"] += 1

    with pytest.raises(
        RuntimeError,
        match=r"frame 000000.*timestamp.*1000001.*expected 1000000",
    ):
        replica._initialize_trajectory(sim, args, [record])


def test_direct_generate_scene_enforces_habitat_022(monkeypatch):
    from data_generation.robot_bev.sources import replica

    monkeypatch.setattr(
        habitat_common,
        "require_habitat_sim",
        lambda: SimpleNamespace(__version__="0.3.3"),
    )
    args = SimpleNamespace(
        allow_version_mismatch=True,
        preflight_only=True,
    )

    with pytest.raises(
        RuntimeError,
        match=r"formal Replica generation requires Habitat-Sim 0\.2\.2",
    ):
        replica.generate_scene(
            args,
            scene_files=SimpleNamespace(),
            scene_split="train",
            writer=SimpleNamespace(),
        )


def test_visualization_options_are_explicitly_unsupported_in_help():
    help_text = make_parser().format_help().lower()

    assert "--save-visualization" in help_text
    assert "--save-ply" in help_text
    assert help_text.count("deprecated and unsupported") >= 2


@pytest.mark.parametrize("option", ("--save-visualization", "--save-ply"))
def test_deprecated_visualization_options_are_rejected(option, monkeypatch):
    args = make_parser().parse_args(
        [
            "--dataset",
            "replica.scene_dataset_config.json",
            "--dataset-id",
            "replica_robot_bev_v4",
            option,
        ]
    )
    monkeypatch.setattr(
        habitat_common,
        "require_habitat_sim",
        lambda: SimpleNamespace(__version__="0.2.2"),
    )

    with pytest.raises(
        ValueError,
        match=r"deprecated and unsupported.*canonical writer",
    ):
        run_generation(args)


def test_direct_generate_scene_rejects_deprecated_visualization(monkeypatch):
    from data_generation.robot_bev.sources import replica

    monkeypatch.setattr(
        habitat_common,
        "require_habitat_sim",
        lambda: SimpleNamespace(__version__="0.2.2"),
    )
    args = SimpleNamespace(
        save_visualization=True,
        save_ply=False,
    )

    with pytest.raises(
        ValueError,
        match=r"deprecated and unsupported.*canonical writer",
    ):
        replica.generate_scene(
            args,
            scene_files=SimpleNamespace(),
            scene_split="train",
            writer=SimpleNamespace(),
        )
