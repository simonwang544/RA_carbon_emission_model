import os

# Define the base directory
base_dir = "/Users/oo/Desktop/carbon_emission_model"

# Define the main directories to be created
main_dirs = ["CEMS_plant", "CEMS_plant_filter"]

# Define the subdirectories for CEMS_plant_filter
sub_dirs = ["CSV", "FIG", "Fitting_fig", "Fitting_result"]

# Define the nested subdirectories within CSV, FIG, and Fitting_fig
nested_sub_dirs = ["P100", "P500", "P1000", "PBig"]

# Function to create directories
def create_directories():
    # Create main directories
    for main_dir in main_dirs:
        main_dir_path = os.path.join(base_dir, main_dir)
        os.makedirs(main_dir_path, exist_ok=True)

        # Create subdirectories for CEMS_plant_filter
        if main_dir == "CEMS_plant_filter":
            for sub_dir in sub_dirs:
                sub_dir_path = os.path.join(main_dir_path, sub_dir)
                os.makedirs(sub_dir_path, exist_ok=True)

                # Create nested subdirectories within CSV, FIG, and Fitting_fig
                if sub_dir in ["CSV", "FIG", "Fitting_fig"]:
                    for nested_sub_dir in nested_sub_dirs:
                        nested_sub_dir_path = os.path.join(sub_dir_path, nested_sub_dir)
                        os.makedirs(nested_sub_dir_path, exist_ok=True)

# Execute the function to create the directories
create_directories()
