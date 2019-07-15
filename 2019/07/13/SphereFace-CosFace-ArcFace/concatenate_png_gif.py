import cv2
import imageio
import numpy as np

def concat_png(file1, file2, outFileName):

    img1 = cv2.imread(file1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(file2, cv2.IMREAD_COLOR)
    img  = np.concatenate([img1, img2], axis=0)

    cv2.imwrite(outFileName, img)

    return img

def concat_gif(modelTypeLists, outFileName):

    angular_gif = ["gif/%s.gif" % mt for mt in modelTypeLists]
    spherized_gif = ["gif/%s_spherized.gif" % mt for mt in modelTypeLists]

    angular_gif_list = list(map(imageio.mimread, angular_gif))
    spherized_gif_list = list(map(imageio.mimread, spherized_gif))

    minfn = min([len(imtype) for imtype in angular_gif_list + spherized_gif_list])

    frames = []

    for i in range(minfn):

        frame = None

        for j in range(len(modelTypeLists)):
            angular_and_spherized = np.concatenate([angular_gif_list[j][i], spherized_gif_list[j][i]], axis=0)
            frame = angular_and_spherized if frame is None else np.concatenate([frame, angular_and_spherized], axis=1)

            for k in range(len(modelTypeLists)):
                cv2.putText(frame, modelTypeLists[k], (600*k + 20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)

        # 打上水印就是我的图了hhhh
        pos = np.array(frame.shape[:-1]) // 2 - np.array([-550, 600]) // 2
        cv2.putText(frame, "louishsu", tuple(pos)[::-1], cv2.FONT_HERSHEY_SIMPLEX, 10, (233, 233, 233), 15)
        frames += [frame]
        # cv2.imshow("frame", cv2.resize(frame, tuple(np.array(frame.shape[:-1]) // 4)[::-1])); cv2.waitKey(0)

    imageio.mimsave(outFileName, frames)

    return frames

if __name__ == "__main__":
    
    modeltype = ['modified', 'cos', 'sphere', 'arc_s1', 'arc_s4', 'arc_s8', 'arc_s16']
    concat_gif(modeltype, 'exp1_dim3.gif')
    
    modeltype = ['cosmul_m2', 'cosmul_m3', 'cosmul_m4', 'adaptive']
    concat_gif(modeltype, 'exp2_dim3.gif')

    # concat_png('gif/start 0.png', 'gif/end 0.png', 'exp2_dim2.png')