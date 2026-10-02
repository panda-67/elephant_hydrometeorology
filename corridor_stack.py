from src.corridor.stack import CorridorPredictorStack


def main():
    print("=" * 60)
    print("P6.6 — CORRIDOR PREDICTOR STACK")
    print("=" * 60)

    stack = CorridorPredictorStack()

    stack.load_master_grid()
    stack.validate_inputs()
    stack.build()
    stack.validate_stack()
    stack.report()

    print("\nP6.6 completed.")


if __name__ == "__main__":
    main()
