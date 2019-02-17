import os
import cv2
import numpy as np
from wordcloud import WordCloud

font = 'C:/Windows/Fonts/SIMYOU.TTF'    # 幼圆
string = 'LouisHsu 单键 小叔叔 想静静 95后 傲娇 skrrrrrrr 大猫座 佛了 要秃 嘤嘤嘤 真香'

mask = cv2.imread('./mask.jpg', cv2.IMREAD_GRAYSCALE)
thresh, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

wc = WordCloud(
        font_path=font, 
        background_color='white',
        color_func=lambda *args, **kwargs: (0,0,0),
        mask=mask,
        max_words=500,
        min_font_size=4,
        max_font_size=None,
        contour_width=1,
        repeat=True                     # 允许词重复
    )
wc.generate_from_text(string)
wc.to_file('./wc.jpg')                  #保存图片