# Notes

generate trajectories using GMAT-Thrust-Data repo

# generating results

latest script:
./generateJournalThrustClass.sh

## parsing results
```
cd gmat/data/classification/
python displayLogData.py . --group-dir {orbit}
```