import importlib
import os
from dotenv import load_dotenv
import subprocess

load_dotenv()

if __name__ == "__main__":
    # Run stock prediction and generate nifty_50_predictions.jsonl
    try:
        # Update the command to run stock_decision_prediction.py script
        cmd = ["python", "examples/prediction model/stock_decision_prediction.py"]

        # Execute the command
        subprocess.run(cmd, check=True)

        print("Successfully generated nifty_50_predictions.jsonl.")
    except subprocess.CalledProcessError:
        print("Script execution failed.")
    except FileNotFoundError:
        print("Python interpreter or the script was not found.")

    # Run Discounts API
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8080))
    app_api = importlib.import_module("examples.api.app")
    app_api.run(host=host, port=port)
