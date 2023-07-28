import matplotlib.pyplot as plt
import cv2
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

config = Cfg.load_config_from_name('vgg_transformer')

config['weights'] = 'E:/Thuc_tap/CCCD/Vietnamese-Id-Card-master/Transformer_OCR/transformerocr.pth'
config['cnn']['pretrained']=False
config['device'] = 'cpu'
config['predictor']['beamsearch']=False

detector = Predictor(config)

def convert_img_to_text(img):
	# predict
	result = detector.predict(img)
	return result