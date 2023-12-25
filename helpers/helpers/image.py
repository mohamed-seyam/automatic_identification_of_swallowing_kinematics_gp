import numpy as np 
from PIL import Image, ImageDraw, ImageFont
import cv2

def draw_bounding_box_on_image( image,
                                ymin,
                                xmin,
                                ymax,
                                xmax,
                                str_to_display,
                                color='blue',
                                thickness=4,
                                use_normalized_coordinates=True):
    """Adds a bounding box to an image.

    Bounding box coordinates can be specified in either absolute (pixel) or
    normalized coordinates by setting the use_normalized_coordinates argument.

    Each string in display_str_list is displayed on a separate line above the
    bounding box in black text on a rectangle filled with the input 'color'.
    If the top of the bounding box extends to the edge of the image, the strings
    are displayed below the bounding box.

    Args:
    image: a PIL.Image object.
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                        (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
        ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
        coordinates as absolute.
    """
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    draw = ImageDraw.Draw(image_pil)
    im_width, im_height = image_pil.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                    ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    if thickness > 0:
        draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
                (left, top)],
                width=thickness,
                fill=color)
    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except IOError:
        font = ImageFont.load_default()

    text_bottom = top

  # Reverse list and print from bottom to top.
    text_width, text_height = font.getsize(str_to_display)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle(
        [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                            text_bottom)],
        fill=color)
    draw.text(
        (left + margin, text_bottom - text_height - margin),
        str_to_display,
        fill='black',
        font=font)
    text_bottom -= text_height - 2 * margin

    np.copyto(image, np.array(image_pil))


def overlay_mask(img, mask, mask_color=[255, 0, 0], alpha=1.0, th=10):
    '''
    img: gray scale img
    mask: to be colored
    color: color of mask
    alpha: opacity
    th: >10 for img [0:255]  >.5 for [0:1]
    '''

    img = img
    rows, cols = img.shape
    # # Construct a colour image to superimpose
    color_mask = np.zeros((rows, cols, 3))
    color_mask[mask > th]  = mask_color  # Red block

    # Construct RGB version of grey-level image
    img_color = np.dstack((img, img, img))

    # Convert the input image and color mask to Hue Saturation Value (HSV)
    # colorspace
    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    # Replace the hue and saturation of the original image
    # with that of the color mask
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)

    return img_masked * 255


def create_distribution_img(video_size, video_data_dir, dynamic_and_static_data_dir, color, overlay_txt, dist_name):
    img = np.ones((video_size, 200, 3)) * 255
    target_regions = pd.read_csv(dynamic_and_static_data_dir)
    for begin, end in zip(target_regions['begin'], target_regions['end']):
        img[begin: end, :] = color

    img = cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
    img = cv2.putText(img, overlay_txt,  (0, int(video_size/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
    img = cv2.resize(img,(400,7),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
    img_path = os.path.join(video_data_dir, dist_name)
    cv2.imwrite(img_path, img)
    return img
