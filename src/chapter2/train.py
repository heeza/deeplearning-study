import sys 
sys.path.append('..')
from common.trainer import Trainer 
from common.optimizer import Adam 
from simple_cbow import SimpleCBOW
from common.util import preprocess, create_contexts_target, convert_one_hot

