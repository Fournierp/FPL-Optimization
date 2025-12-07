import csv
from pathlib import Path


def convert_txt_to_csv(input_file: str, output_file: str) -> None:
    with Path(input_file).open('r') as f:
        lines = f.readlines()

    if len(lines) == 0:
        msg = 'Input file is empty'
        raise ValueError(msg)

    column_names = [col.strip() for col in lines[0].split('\t') if col.strip()]

    # Parse the data - pattern is: blank line, name, position/price, data line
    data_rows = []
    i = 1  # Skip the header line at index 0

    # Remove all blank lines upfront
    lines = [line for line in lines if line.strip()]

    while i < len(lines):
        line = lines[i].strip()

        player_name = line.strip()

        # Next line should be position and price
        if i + 1 >= len(lines):
            break
        position_price_line = lines[i + 1].strip()
        position_and_price = position_price_line.split()

        # Next line should contain position and price
        if len(position_and_price) <= 1:
            i += 1
            continue

        position = position_and_price[0]
        price = position_and_price[1]

        # Next line should be the projection data
        if i + 2 >= len(lines):
            break
        projection_data = lines[i + 2].strip().split()

        # Combine: name, position, price, then all data columns
        row = [player_name, position, price, *projection_data]
        data_rows.append(row)

        i += 3  # Move to the next player

    with Path(output_file).open('w', newline='') as f:
        writer = csv.writer(f)
        header = ['NAME', 'POSITION', 'PRICE', *column_names]
        writer.writerow(header)
        writer.writerows(data_rows)

    print(f'✓ Successfully converted {len(data_rows)} players to CSV')
    print(f'  Input:  {input_file}')
    print(f'  Output: {output_file}')

