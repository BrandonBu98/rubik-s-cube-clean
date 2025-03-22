import cv2
import docdetect
import numpy as np
from matplotlib import pyplot as plt
import time

# # print(docdetect.__file__)
#
def docdetect_image_run():
    # image = cv2.imread('./cube_test_data/test20.jpg') 
    # image = cv2.imread('./good_frames/good_cube190.jpg') 
    image = cv2.imread('./good_frames/good_cube9.jpg') 
    rects = docdetect.process(image)
    image = docdetect.draw(rects, image)
    plt.subplot(121)
    plt.imshow(image)
    plt.title('docdetect output')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    #
    # for i in range(200):
    #     image = cv2.imread('./good_frames/good_cube190.jpg') 
    #     image = cv2.imread(f'./good_frames/good_cube{i+1}.jpg') 
    #     rects = docdetect.process(image)
    #     image = docdetect.draw(rects, image)
    #     plt.subplot(121)
    #     plt.imshow(image)
    #     plt.title('docdetect output')
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.show()
    #     time.sleep(0.5)

    # cv2.imshow('output', image)
    # cv2.waitKey(10)

def docdetect_video_run():
    video = cv2.VideoCapture(".\\cube_raw_footage\\normal_cube.mp4")
    cv2.startWindowThread()
    cv2.namedWindow('output')
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            rects = docdetect.process(frame)
            frame = docdetect.draw(rects, frame)
            cv2.imshow('output', frame)
            cv2.waitKey(1)
    video.release()

def edge_detect_run():

    img = cv2.imread('./test-cube.png') 
    # img = cv2.imread('./cube_test_data/test20.jpg') 
    assert img is not None, "file could not be read, check with os.path.exists()"
    edges = cv2.Canny(img,200,200)

    cv2.imshow('original', img)
    cv2.imshow('edges', edges)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Added to test contours
    # contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # # cv2.imshow('Canny Edges After Contouring', edges) 
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    # cv2.imshow('Contours', img)
    # if cv2.waitKey(0) & 0xff == 27: 
    #     cv2.destroyAllWindows()
     
    # plt.subplot(121)
    # plt.imshow(img,cmap = 'gray')
    # plt.title('Original Image')
    # plt.xticks([])
    # plt.yticks([])
    # plt.subplot(122)
    # plt.imshow(edges,cmap = 'gray')
    # plt.title('Edge Image')
    # plt.xticks([])
    # plt.yticks([])
    #  
    # plt.show()

def corner_detect_run():
    img = cv2.imread('test-cube.png') 
    # img = cv2.imread('./cube_test_data/test20.jpg') 
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray) 
    dest = cv2.cornerHarris(gray, 2, 3, 0.04)  # TEST:
    dest = cv2.dilate(dest, None) 
  
    # Reverting back to the original image, 
    # with optimal threshold value 
    img[dest > 0.01 * dest.max()]=[0, 0, 255] 
      
    # the window showing output image with corners 
    cv2.imshow('Image with Borders', img) 
      
    # De-allocate any associated memory usage  
    if cv2.waitKey(0) & 0xff == 27: 
        cv2.destroyAllWindows()
    # plt.subplot(121)
    # plt.imshow(img,cmap = 'gray')
    # plt.title('Original Image')
    # plt.xticks([])
    # plt.yticks([])
    # plt.subplot(122)
    # plt.imshow(corners,cmap = 'gray')
    # plt.title('Corner Image')
    # plt.xticks([])
    # plt.yticks([])
     
    plt.show()

def contours_detect_run():
    img = cv2.imread('test-cube.png') 
    # img = cv2.imread('./cube_test_data/test20.jpg') 
    assert img is not None, "file could not be read, check with os.path.exists()"
    edges = cv2.Canny(img,200,200)

    # Added to test contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.imshow('Canny Edges After Contouring', edges) 
    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    cv2.imshow('Contours', img)
    if cv2.waitKey(0) & 0xff == 27: 
        cv2.destroyAllWindows()

if __name__ == "__main__":
    contours_detect_run()
    # corner_detect_run()
    # edge_detect_run()
    # docdetect_image_run()
    # docdetect_video_run()
