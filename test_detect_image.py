import cv2
from lp_image_detect import recognize_lp_from_image

for i in range(301):
    _frame_name = 'frame%d.jpg' % i
    path = 'Data/images/outImages3/' + _frame_name
    # read image from path
    _input_img = cv2.imread(path)
    # read plate from image
    image = recognize_lp_from_image(_input_img)
    cv2.imwrite('Outputs/images/correctImages3/%s' % _frame_name, image)

cv2.destroyAllWindows()