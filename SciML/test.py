from loss_funktion.bout_info import BOUTHESELInfo

from pathlib import Path

root = Path(__file__).resolve().parents[1] / "simulatorer" / "BOUT" / "BOUT-HESEL" / "data"
info = BOUTHESELInfo(root)

# Sort parameters by name and print them
sorted_parameters = sorted(info.parameters.items())

for navn, værdi in sorted_parameters:
    print(f"{navn}: {værdi}")