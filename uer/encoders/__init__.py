from uer.encoders.transformer_encoder_Vertical import TransformerEncoder_Vertical
from uer.encoders.transformer_encoder import TransformerEncoder
from uer.encoders.rnn_encoder import RnnEncoder
from uer.encoders.rnn_encoder import LstmEncoder
from uer.encoders.rnn_encoder import GruEncoder
from uer.encoders.rnn_encoder import BirnnEncoder
from uer.encoders.rnn_encoder import BilstmEncoder
from uer.encoders.rnn_encoder import BigruEncoder
from uer.encoders.dual_encoder import DualEncoder
from uer.encoders.cnn_encoder import GatedcnnEncoder
from uer.encoders.cnn_encoder import PlaincnnEncoder
from uer.encoders.cnn_encoder import DepthwisecnnEncoder
from uer.encoders.cnn_encoder import LightweightcnnEncoder
from uer.encoders.cnn_encoder import DynamiccnnEncoder
from uer.encoders.cnn_encoder import GatedLightweightCNNEncoder
from uer.encoders.cnn_encoder import GatedLightweightCNNEncoder_0707
from uer.encoders.cnn_encoder import GatedLightweightCNNEncoder_0709
from uer.encoders.cnn_encoder import GatedLightweightCNNEncoder_0707_NoSoftmax
from uer.encoders.cnn_encoder import GatedLightweightCNNEncoder_0707_NoGate
from uer.encoders.cnn_encoder import GatedLightweightCNNEncoder_0707_ReLU
from uer.encoders.cnn_encoder import GatedLightweightCNNEncoder_0707_GeLU


str2encoder = {"transformer_Vertical": TransformerEncoder_Vertical,"transformer": TransformerEncoder, "rnn": RnnEncoder, "lstm": LstmEncoder,
               "gru": GruEncoder, "birnn": BirnnEncoder, "bilstm": BilstmEncoder, "bigru": BigruEncoder, "dual": DualEncoder,
               "gatedcnn": GatedcnnEncoder, "plaincnn": PlaincnnEncoder, "depthwisecnn": DepthwisecnnEncoder, "lightweightcnn": LightweightcnnEncoder, "dynamiccnn": DynamiccnnEncoder,
               "gatedlightweightcnn": GatedLightweightCNNEncoder, "gatedlightweightcnn_0707": GatedLightweightCNNEncoder_0707,  "gatedlightweightcnn_0709": GatedLightweightCNNEncoder_0709,
               "gatedlightweightcnn_0707_nosoftmax": GatedLightweightCNNEncoder_0707_NoSoftmax, "gatedlightweightcnn_0707_nogate": GatedLightweightCNNEncoder_0707_NoGate, 
               "gatedlightweightcnn_0707_relu": GatedLightweightCNNEncoder_0707_ReLU, "gatedlightweightcnn_0707_gelu": GatedLightweightCNNEncoder_0707_GeLU}

__all__ = ["TransformerEncoder_Vertical", "TransformerEncoder", "RnnEncoder", "LstmEncoder", "GruEncoder", "BirnnEncoder",
           "BilstmEncoder", "BigruEncoder", "DualEncoder", "str2encoder",
           "GatedcnnEncoder", "PlaincnnEncoder", "DepthwisecnnEncoder", "LightweightcnnEncoder", "DynamiccnnEncoder",
           "GatedLightweightCNNEncoder", "GatedLightweightCNNEncoder_0707", "GatedLightweightCNNEncoder_0709",
           "GatedLightweightCNNEncoder_0707_NoSoftmax", "GatedLightweightCNNEncoder_0707_NoGate", "GatedLightweightCNNEncoder_0707_ReLU", "GatedLightweightCNNEncoder_0707_GeLU"]
