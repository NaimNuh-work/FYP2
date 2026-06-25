9import sys


def dms_to_decimal(degrees, minutes, seconds, hemisphere):
    """Mathematical conversion from DMS to Decimal Degrees."""
    decimal_degrees = float(degrees) + (float(minutes) / 60.0) + (float(seconds) / 3600.0)

    # Apply the Hemisphere Rule
    if hemisphere.upper() in ['S', 'W']:
        decimal_degrees = -decimal_degrees

    return decimal_degrees


def main():
    print("========================================")
    print("  FAST GPS TRANSLATOR (Continuous Loop) ")
    print("========================================")
    print("Format:  Degrees Minutes Seconds Hemisphere")
    print("Example: 3 0 32 N")
    print("Type 'q' and hit Enter to quit.\n")

    # This 'while True' creates the infinite loop
    while True:
        try:
            # 1. Ask for everything on one line
            user_input = input("Enter coordinate: ").strip()

            # Check if you want to quit
            if user_input.lower() == 'q':
                break

            # 2. Split the input into 4 pieces
            parts = user_input.split()

            if len(parts) != 4:
                print("[ERROR] Please enter exactly 4 parts separated by spaces (e.g., 3 0 32 N).\n")
                continue  # Jumps back to the start of the loop

            degrees = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            hemisphere = parts[3].upper()

            if hemisphere not in ['N', 'S', 'E', 'W']:
                print("[ERROR] Hemisphere must be N, S, E, or W.\n")
                continue  # Jumps back to the start of the loop

            # 3. Calculate and print instantly
            result = dms_to_decimal(degrees, minutes, seconds, hemisphere)
            print(f">>> Result: {result:.6f}\n")

        except ValueError:
            print("[ERROR] Make sure you are using numbers for degrees, minutes, and seconds.\n")

    print("\n[INFO] Translator shut down successfully.")


if __name__ == "__main__":
    main()