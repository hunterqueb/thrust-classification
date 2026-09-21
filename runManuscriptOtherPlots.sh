#!/bin/sh

python scripts/two_body/analyzeThrustSeparability.py --orbit leo --plot --fontScale 1.5
python scripts/two_body/analyzeThrustSeparability.py --orbit meo --plot --fontScale 1.5
python scripts/two_body/analyzeThrustSeparability.py --orbit geo --plot --fontScale 1.5