from src.corridor.boundary import CorridorBoundary


def main():
    print("=" * 60)
    print("CORRIDOR BOUNDARY")
    print("=" * 60)

    boundary = CorridorBoundary()

    boundary.load()
    boundary.prepare()
    boundary.build()
    boundary.save()
    boundary.report()

    print("\nBoundary completed.")


if __name__ == "__main__":
    main()
