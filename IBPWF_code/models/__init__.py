from .factorized_ibp_mlp import MLP_IBP_WF
from .factorized_ibp_cnn import ResNet_IBP_WF
from .factorized_ibp_wide_resnet import WideResNet_IBP_WF

MODEL = {
		'factorized_ibp_mlp': MLP_IBP_WF,
		'factorized_ibp_cnn': ResNet_IBP_WF,
		'ResNet20': ResNet_IBP_WF,
		'WideResNet': WideResNet_IBP_WF,
}