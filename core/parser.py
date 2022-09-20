import argparse

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--data-dir'   , type=str  , required=True)
parser.add_argument('--log-dir'    , type=str  , required=True)
parser.add_argument('--log-desc'   , type=str  , required=True)
parser.add_argument('--batch-size' , type=int  , required=True)
parser.add_argument('--model'      , type=str  , required=True)
parser.add_argument('--saved-model', type=str  , default=None)
parser.add_argument('--num-epochs' , type=int  , required=True)
parser.add_argument('--lr'         , type=float, required=True)

args = parser.parse_args()