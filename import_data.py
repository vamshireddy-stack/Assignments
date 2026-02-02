import kagglehub
from kagglehub import KaggleDatasetAdapter

file_path = "car_data.csv"

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "nehalbirla/vehicle-dataset-from-cardekho",
  file_path
)
