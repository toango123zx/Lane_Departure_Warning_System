# import cv2
# import numpy
# import matplotlib.pyplot

# width = 1640
# height = 590

# def Show_camera(ret, frame):
#     # polygons = numpy.array([[(175, 302), (388, 302), (269, 205), (257, 205)]])
#     # # polygons = numpy.array([[(415, 535), (415, 1185), (280, 822), (280, 788)]])
#     # mask = numpy.zeros_like(frame)
#     # cv2.fillPoly(mask, polygons, 255)
#     # # frame = cv2.bitwise_and(frame, mask)
#     # frame =cv2.addWeighted(frame, 0.8, mask, 1, 1)
#     frame = cv2.resize(frame, (width, height))
#     # if ret == True:
#     #     cv2.imshow('Video', frame)
#     return frame


# # Khởi tạo đối tượng VideoCapture để truy cập camera
# cap = cv2.VideoCapture(2)

# # Thiết lập kích thước mặc định cho camera
# # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 590)
# # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1640)

# # Tiếp tục đọc frame và xử lý như thông thường
# while True:
#     ret, frame = cap.read()
#     frame = cv2.flip(frame, 1) 
#     # print(ret)
#     fig, axs = matplotlib.pyplot.subplots(1, 2)
#     axs[0].imshow(frame)
#     img1 = Show_camera(ret, frame)
#     axs[1].imshow(img1)
#     matplotlib.pyplot.show()
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # Giải phóng tài nguyên và đóng cửa sổ
# cap.release()
# cv2.destroyAllWindows()


# import cv2

# def get_available_webcams():
#     available_webcams = []
#     i = 0
#     while True:
#         cap = cv2.VideoCapture(i)
#         if not cap.read()[0]:
#             break
#         else:
#             available_webcams.append(i)
#         cap.release()
#         i += 1
#     return available_webcams

# # Lấy danh sách webcam hiện có
# webcams = get_available_webcams()

# # Hiển thị danh sách webcam
# print("Danh sách webcam:")
# for webcam in webcams:
#     print(f"Webcam {webcam}")

import cv2
import numpy

width = 1640
height = 590

def Show_camera(ret, frame):
    polygons = numpy.array([[(535, 415), (1185, 415), (822, 280), (788, 280)]])
    mask = numpy.zeros_like(frame)
    cv2.fillPoly(mask, polygons, 255)
    result = cv2.bitwise_and(frame, mask)
    result =cv2.addWeighted(frame, 0.8, mask, 1, 1)
    if ret == True:
        cv2.imshow('Video', result)

def read_webcam_image():
    camera1 = cv2.VideoCapture(1)
    camera1.set(cv2.CAP_PROP_FPS, 60)
    
    # camera3 = cv2.VideoCapture(2)
    # camera3.set(cv2.CAP_PROP_FPS, 60)
      # Số 0 thể hiện webcam đầu tiên

    while True:
        ret, frame = camera1.read()
        frame = cv2.resize(frame, (width, height))
        Show_camera(ret, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng tài nguyên và đóng cửa sổ hiển thị
    camera1.release()
    # camera3.release()
    cv2.destroyAllWindows()

# Đọc hình ảnh từ webcam
read_webcam_image()
