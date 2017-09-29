'''
TESTS : OpenEXR utilities functions
'''

import argparse
import os
import sys
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

    exrChnRecords.printExrInfo(args.input)
    exrChnRecords.writeExrChannels(
        args.input, args.output, args.output_channels)
    exrChnRecords.printExrInfo(args.output)

    print exrChnRecords.readExrData(args.output, args.output_channels).shape

    return 0

#-------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
