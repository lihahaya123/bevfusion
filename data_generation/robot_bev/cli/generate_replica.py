"""Generate canonical robot BEV data from original Replica v1 scenes."""

from data_generation.robot_bev.sources.replica import make_parser, run_generation


def main() -> None:
    args = make_parser().parse_args()
    run_generation(args)


if __name__ == "__main__":
    main()
