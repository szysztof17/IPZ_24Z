import os
import re
import csv
import argparse


# Function to parse a single log file and extract ligand ID from the filename
def parse_vina_log(file_path):
    results = []
    
    # Extract the ligand ID from the filename (assuming it is part of the filename)
    ligand_id = os.path.basename(file_path).split('.')[0]  # Extract before the first dot

    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Regular expression pattern to capture the information we need
    pattern = r"^\s*(\d+)\s+([-]?\d+\.\d+)\s+([\d\.]+)\s+([\d\.]+)"
    
    # Go through each line and extract data using the regex pattern
    for line in lines:
        match = re.match(pattern, line)
        if match:
            mode = int(match.group(1))
            affinity = float(match.group(2))
            rmsd_lb = float(match.group(3))  # lower bound of RMSD
            rmsd_ub = float(match.group(4))  # upper bound of RMSD
            # Append the ligand ID with the docking data
            results.append([ligand_id, mode, affinity, rmsd_lb, rmsd_ub])

    return results

# Function to parse all log files in the directory
def parse_all_vina_logs(log_dir):
    all_results = []
    
    for filename in os.listdir(log_dir):
        if filename.endswith('.log'):
            file_path = os.path.join(log_dir, filename)
            file_results = parse_vina_log(file_path)
            all_results.extend(file_results)
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description="Parse Vina log files and save results to a CSV file.")
    parser.add_argument("log_dir", help="Directory containing the Vina log files.")
    parser.add_argument("output_csv", help="Path to the output CSV file.")

    args = parser.parse_args()

    # Parse the log files and store the results
    results = parse_all_vina_logs(args.log_dir)

    # Save the results to the specified CSV file
    with open(args.output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Ligand ID', 'Mode', 'Affinity (kcal/mol)', 'RMSD Lower Bound', 'RMSD Upper Bound'])
        writer.writerows(results)

    print(f'Parsed data has been saved to {args.output_csv}')

if __name__ == "__main__":
    main()

