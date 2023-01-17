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
parser.add_argument('--optimizer'  , type=str  , default='SGD')
parser.add_argument('--sched_pct'  , type=float, default=0.025)

args = parser.parse_args()