import cv2
import numpy as np

def IOU(a, b):
    """ intersection over union
    Params:
        a, b: {list(x1, y1, x2, y2)}
    Returns:
        iou: {float}
    """
    x1a, y1a, x2a, y2a = a
    x1b, y1b, x2b, y2b = b
    _x1 = max(x1a, x1b)
    _y1 = max(y1a, y1b)
    _x2 = min(x2a, x2b)
    _y2 = min(y2a, y2b)
    _h, _w = abs(_y2 - _y1), abs(_x2 - _x1)
    i = _h * _w 
    u =  (x2a - x1a) * (y2a - y1a) + (x2b - x1b) * (y2b - y1b) - i
    iou = i / u
    return iou

dog  = [ 95, 295, 140, 250, (  0,   0, 255)]
bike = [205, 212, 340, 214, (  0, 255, 255)]
car  = [365,  92, 150,  64, (255, 255,   0)]
anchors = [[2.5, 1.5], [1.5, 2.5]]
resolutions = [4, 6, 8]

image = cv2.imread('resized.jpg', cv2.IMREAD_COLOR)
H, W = image.shape[:2]

for r in resolutions:
    im = image.copy()

    cellsize = H // r

    # 网格
    for i in range(r):
        cv2.line(im, (cellsize * i, 0), (cellsize * i, H), (255, 255, 255))
        cv2.line(im, (0, cellsize * i), (W, cellsize * i), (255, 255, 255))

    # ground truth
    for x, y, w, h, c in [dog, bike, car]:
        cv2.rectangle(im, (x - w // 2, y - h // 2), 
                                (x + w // 2, y + h // 2), c, 2)

        # anchor
        for i in range(r):
            for j in range(r):
                
                xa = int((i + 0.5) * cellsize)
                ya = int((j + 0.5) * cellsize)

                for anchor in anchors:
                    ha = int(cellsize * anchor[0])
                    wa = int(cellsize * anchor[1])
                    _x1 = max(0, xa - wa // 2)
                    _y1 = max(0, ya - ha // 2)
                    _x2 = min(H, xa + wa // 2)
                    _y2 = min(W, ya + ha // 2)
                    iou = IOU([x - w // 2, y - h // 2, x + w // 2, y + h // 2], [_x1, _y1, _x2, _y2])
                    if iou > 0.5 and \
                        xa > x - w // 2 and xa < x + w // 2 \
                            and ya > y - h // 2 and ya < y + h // 2:
                        cv2.rectangle(im, (xa - wa // 2, ya - ha // 2), 
                                                (xa + wa // 2, ya + ha // 2), c, 1)
                        cv2.circle(im, (xa, ya), 4, c, -1)

    cv2.imshow("im%dx%d" % (r, r), im)
    cv2.imwrite("im%dx%d_anchor.jpg" % (r, r), im)

cv2.waitKey(0)