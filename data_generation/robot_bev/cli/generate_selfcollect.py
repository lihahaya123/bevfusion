"""Generate canonical Robot BEV data from self-collected left-camera data."""

from data_generation.robot_bev.sources.selfcollect import make_parser, run_generation


def main() -> None:
    run_generation(make_parser().parse_args())


if __name__ == "__main__":
    main()
