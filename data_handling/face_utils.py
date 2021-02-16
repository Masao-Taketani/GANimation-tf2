import os
import subprocess
from glob import glob

from PIL import Image
import face_recognition


IMG_SIZE = 128

def detect_faces(img_path):
    img = face_recognition.load_image_file(img_path)
    # face_los: [(top, right, bottm, left)]
    face_loc = face_recognition.face_locations(img)
    return face_loc


def crop_and_save_faces(img_path, face_loc, to_dir):
    bname = os.path.basename(img_path)
    pil_img = Image.open(img_path)
    top, right, bottom, left = face_loc
    pil_img_crop = pil_img.crop((left, top, right, bottom))
    pil_img_crop = pil_img_crop.resize((IMG_SIZE, IMG_SIZE))
    pil_img_crop.save(os.path.join(to_dir, bname), quality=95)


def get_aus(openface_dir, img_dir):
    exe = os.path.join(openface_dir, "build/bin/FaceLandmarkImg")
    command = [exe, "-fdir", img_dir, "-aus"]
    subprocess.run(command)


def detect_crop_and_save_faces(img_dir, cropped_dir):
    img_list = glob(os.path.join(img_dir, "*.jpg"))
    for img in img_list:
        face_loc = detect_faces(img)
        if len(face_loc) == 1:
            crop_and_save_faces(img, face_loc[0], cropped_dir)



if __name__ == "__main__":
    test_dir = "../tmp/test/"
    cropped_dir = "../tmp/cropped/"
    openface_dir = "../../../OpenFace/"
    detect_crop_and_save_faces(test_dir, cropped_dir)
    get_aus(openface_dir, cropped_dir)