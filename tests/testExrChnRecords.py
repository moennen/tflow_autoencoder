'''
TESTS : OpenEXR utilities functions
'''

import argparse
import os
import sys
import sys
import os
import tensorflow as tf
sys.path.append(os.path.abspath(
    '/mnt/p4/avila/moennen_wkspce/tflow_autoencoder'))
import exrChnRecords

#-------------------------------------------------------------------------


def main():
    #-------------------------------------------------------------------------
    # ArgumentParser
    parser = argparse.ArgumentParser()
    # input images sequences
    parser.add_argument(
        "input",
        help="input openEXR filename"
    )
    parser.add_argument(
        "output",
        help="output openEXR filename"
    )
    parser.add_argument(
        "output_channels",
        help="output channels"
    )
    args = parser.parse_args()

    printOpenEXRInfo(args.input)
    convertOpenEXR(args.input, args.output, args.output_channels)
    printOpenEXRInfo(args.output)

    print getOpenEXR(args.output)

    return 0

#-------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
