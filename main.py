import requests
import base64
import os
import sys
import json
import cv2
import numpy as np
import time
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import io
import moviepy.editor as mp
import ffmpeg
import settings

API_KEY = settings.API_KEY
API_SECRET = settings.API_SECRET


def get_face_feature(img_file_path):
    """Post image file to Face++ & Get face attributes data, save it to ./json_dumps/"""
    with open(img_file_path, 'rb') as f:
        img_file = base64.encodebytes(f.read())
    config = {'api_key': API_KEY,
              'api_secret': API_SECRET,
              'image_base64': img_file,
              'return_landmark': 1,
              'return_attributes': 'gender,age,smiling,headpose,facequality,blur,eyestatus,emotion,ethnicity,beauty,mouthstatus,eyegaze,skinstatus'
              }
    url = 'https://api-us.faceplusplus.com/facepp/v3/detect'
    res = requests.post(url, data=config)
    data = json.loads(res.text)

    os.makedirs('json_dumps', exist_ok=True)
    base_path = os.path.join('./json_dumps', '')
    basename_without_ext = os.path.splitext(os.path.basename(img_file_path))[0]

    with open(base_path + basename_without_ext + '.json', 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def get_concat_v_blank(im1, im2, color=(0, 0, 0)):
    """Get two images join verticaly"""
    dst = Image.new('RGB', (max(im1.width, im2.width),
                    im1.height + im2.height), color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def extract_audio(video, path):
    """ Extract audio from input video."""
    clip_input = mp.VideoFileClip(video).subclip()
    clip_input.audio.write_audiofile(path + 'audio.mp3')


def plot_data_to_movie(movie):
    """ Extract audio from input video."""
    os.makedirs('output_file', exist_ok=True)
    output_path = os.path.join('./output_file', '')
    extract_audio(movie, output_path)
    """Capture the movie data & get movie's basic info"""
    cap = cv2.VideoCapture(movie)
    if not cap.isOpened():
        exit()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    """ Save all frame image to ./slice/ & post the image individually to Face++"""
    print("Slice movie & Post data to Face++ Start...")
    os.makedirs('slice', exist_ok=True)
    image_path = os.path.join('./slice', '')
    digit = len(str(int(count)))

    n = 0
    progressbar = tqdm(total=count)
    while True:
        ret, frame = cap.read()

        if ret:
            cv2.imwrite('{}{}.{}'.format(
                image_path, str(n).zfill(digit), 'jpg'), frame)
            get_face_feature("{}{}.{}".format(
                image_path, str(n).zfill(digit), 'jpg'))

            n += 1
            progressbar.update(1)
            time.sleep(3)
        else:
            print("Completed!")
            break
    progressbar.close()

    """Get eye_geze(x,y) from ./json_dumps"""
    print("Face data preprocessing Start...")
    json_path = os.path.join('./json_dumps', '')
    n = 0
    digit = len(str(int(count)))

    result_left = pd.DataFrame()
    result_right = pd.DataFrame()
    cols = ['position_x_coordinate', 'position_y_coordinate',
            'vector_x_component', 'vector_y_component', 'vector_z_component']

    progressbar = tqdm(total=count)
    while True:
        df = pd.read_json(json_path + str(n).zfill(digit) + '.json')

        if df.empty:
            df_left = pd.Series(
                [np.nan, np.nan, np.nan, np.nan, np.nan], index=cols)
            df_right = pd.Series(
                [np.nan, np.nan, np.nan, np.nan, np.nan], index=cols)
            result_left = result_left.append(df_left, ignore_index=True)
            result_right = result_right.append(df_right, ignore_index=True)
        else:
            df_left = pd.DataFrame(df["faces"][0]["attributes"]["eyegaze"]["left_eye_gaze"].values(
            ), index=df["faces"][0]["attributes"]["eyegaze"]["left_eye_gaze"].keys()).T
            df_right = pd.DataFrame(df["faces"][0]["attributes"]["eyegaze"]["right_eye_gaze"].values(
            ), index=df["faces"][0]["attributes"]["eyegaze"]["right_eye_gaze"].keys()).T
            result_left = result_left.append(df_left)
            result_right = result_right.append(df_right)
        n += 1
        progressbar.update(1)

        if n == count:
            progressbar.close()
            os.makedirs('csvs', exist_ok=True)
            result_left.to_csv('./csvs/left_eye.csv')
            result_right.to_csv('./csvs/right_eye.csv')
            print("Completed!")
            break

    n = 0
    min_range = 0
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100,
                     facecolor='w', linewidth=0, edgecolor='w')
    axes = fig.subplot_mosaic(
        [['A'],
         ['B']],
        subplot_kw=dict(facecolor='white'))
    background_img = fig2img(fig)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(output_path + 'output_with_graph.m4v', fourcc,
                          fps, (width, height + background_img.height))

    print("Plotting Graph...")
    progressbar = tqdm(total=count)
    while True:
        ret, frame = cap.read()
        if ret:
            fig = plt.figure(figsize=(12.8, 6.4), dpi=100,
                             facecolor='w', linewidth=0, edgecolor='w')
            axes = fig.subplot_mosaic(
                [['A'],
                 ['B']],
                subplot_kw=dict(facecolor='white'))
            min_range = n - 600 if n > 600 else 0
            axes['A'].plot(result_left.query('@min_range <= index & index <= @n').vector_x_component, label='Horizontal',
                           color='r', linewidth=1, linestyle='solid')
            axes['A'].plot(result_left.query('@min_range<=index & index <= @n').vector_y_component, label='Vertical',
                           color='g', linewidth=1, linestyle='solid')
            axes['B'].plot(result_right.query('@min_range<=index & index <= @n').vector_x_component, label='Horizontal',
                           color='r', linewidth=1, linestyle='solid')
            axes['B'].plot(result_right.query('@min_range<=index & index <= @n').vector_y_component, label='Vertical',
                           color='g', linewidth=1, linestyle='solid')
            axes['A'].set_xlabel('Left eye')
            # axes['A'].set_ylabel('vector')
            axes['B'].set_xlabel('Right eye')
            # axes['B'].set_ylabel('vector')
            axes['A'].legend(bbox_to_anchor=(1, 1), loc='upper right',
                             borderaxespad=0, fontsize=14)
            axes['A'].set_ylim([-1.0, 1.0])
            axes['A'].set_yticks([-1.0, -0.8, -0.6, -0.4, -0.2,
                                  0, 0.2, 0.4, 0.6, 0.8, 1.0])
            axes['B'].set_ylim([-1.0, 1.0])
            axes['B'].set_yticks([-1.0, -0.8, -0.6, -0.4, -0.2,
                                  0, 0.2, 0.4, 0.6, 0.8, 1.0])

            buf = io.BytesIO()
            fig.savefig(buf)
            buf.seek(0)
            background_img = Image.open(buf)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            result_image = get_concat_v_blank(pil_image, background_img)
            rgb_image = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)

            out.write(rgb_image)

            n += 1
            plt.clf()
            plt.close()
            progressbar.update(1)
            buf.close()

        if n == count:
            progressbar.close()
            print("Completed!")
            break

    cap.release()
    out.release()

    video = ffmpeg.input(output_path + 'output_with_graph.m4v').video
    audio = ffmpeg.input(output_path + 'audio.mp3').audio
    stream = ffmpeg.output(video, audio, output_path +
                           'output_with_graph_audio.m4v')
    ffmpeg.run(stream)


if __name__ == '__main__':
    args = sys.argv
    if len(args) == 1:
        print('Please input movie data!')
    elif len(args) == 2:
        print(
            'Processing {}...Maybe, It takes more than 150 x (movie time)'.format(args[1]))
        plot_data_to_movie(args[1])
        print('All Process Completed!')
    else:
        print('Too many arguments! Please choose a movie file(mp4,mov)')
