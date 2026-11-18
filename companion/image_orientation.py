"""Rotate inference input and restore box edges to unchanged camera coordinates."""
from .gimbal import valid_face


def checked_rotation(value):
    if type(value) is not int or value not in (0, 90, 180, 270):
        raise ValueError('Analysis rotation must be 0, 90, 180 or 270 degrees clockwise')
    return value


def rotate_frame(frame, rotation):
    checked_rotation(rotation)
    if rotation == 0:
        return frame
    import cv2
    codes = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
             270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    return cv2.rotate(frame, codes[rotation])


def restore_box(box, rotation, width, height):
    checked_rotation(rotation)
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError('Positive original image dimensions required')
    rw, rh = (height, width) if rotation in (90, 270) else (width, height)
    if not valid_face(box, rw, rh):
        raise ValueError('Invalid inference box')
    x1, y1, x2, y2 = box
    if rotation == 90:
        return y1, height-x2, y2, height-x1
    if rotation == 180:
        return width-x2, height-y2, width-x1, height-y1
    if rotation == 270:
        return width-y2, x1, width-y1, x2
    return tuple(box)
