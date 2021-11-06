from numpy.core.fromnumeric import size
import pandas as pd
import typing as t
from ast import literal_eval
import albumentations as a
import random
import os, glob
import numpy as np
from PIL import Image, ImageDraw
import cv2
import re
from functools import partial
import matplotlib.pyplot as plt


LABEL = {'lat' : 1, 'ext' : 2, 'int_a' : 3, 'int_b' : 4}

def _prepare_dataset(df: pd.DataFrame, full_path) -> pd.DataFrame:

    fn_img_path = partial(_get_file_name, full_path)

    cleaning_fn = _chain(
        [
            _drop_useless,
            _string_to_list,
            fn_img_path
        ]
    )
    df = cleaning_fn(df)
    return df

def _chain(functions: t.List[t.Callable[[pd.DataFrame], pd.DataFrame]]):
    def helper(df):
        for fn in functions:
            df = fn(df)
        return df

    return helper

def _drop_useless(df):
   return df.drop(['id', 'annotator', 'annotation_id'], axis = 1)

def _string_to_list(df: pd.DataFrame):  
    new_dict = df.label.apply(literal_eval)
    df['label'] = pd.Series(new_dict)
    return df

def _preprocess_image(img_size, img_path):
    #Resize
    size =(img_size, img_size)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(image, size, interpolation= cv2.INTER_AREA)

    return img_resized

def _regex_search(img_name):
    pattern = re.compile(r"/(\d+)")
    matches = pattern.finditer(img_name)

    for match in matches:
        file_name = match.group() + ".png"

    return file_name

def _get_file_name(full_path, dataset: pd.DataFrame):
    dataset['image'] = full_path + dataset.image.apply(_regex_search)
    return dataset

def build_sources(data_dir, image_size = 512,mode = 'train', gray = False):
    #Debe retornar un dataframe con la estrustura[path_img, img, mask]
    datasets_names = os.listdir(data_dir)
    clean_ann = pd.DataFrame()
    for dataset in datasets_names:
        print(dataset)
        full_dataset_path = os.path.join(data_dir, dataset)
        ann_list = [ file for file in os.listdir(
            full_dataset_path) if file.endswith('csv')]
            
        for ann in ann_list:
            ann_df = pd.read_csv(os.path.join(full_dataset_path, ann), sep=';')
            clean_aux = _prepare_dataset(ann_df, full_dataset_path)
            fn_pre = partial(_preprocess_image, image_size)
            clean_aux['img'] = clean_aux.image.apply(fn_pre)
            clean_aux['mask'] = list(_create_masks(clean_aux, image_size).values())
        clean_ann = clean_ann.append(clean_aux)

    if gray:
        clean_ann['img'] = clean_ann.img.apply(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY))
    return clean_ann

def im_show_three(dataset: pd.DataFrame, title = True):
    fig, ax = plt.subplots(3,2, figsize=(100,100))
    rnd_sample_df = dataset.sample(n=3, random_state=np.random.RandomState())

    for df_sample, row_figure in zip(rnd_sample_df.iterrows(), ax):
        if title:
            row_figure[0].set_title(df_sample[1]['image'], size=100)
        row_figure[0].imshow(df_sample[1].img)
        row_figure[1].imshow(df_sample[1]['mask'])

    fig.tight_layout()
    plt.show()


def _get_transformations():

    transformations = [
        a.Compose([
            a.HorizontalFlip(p = 0.5),
            a.RandomBrightnessContrast(p = 0.3)
        ]),

        a.Compose([
            a.RandomBrightnessContrast(),    
            a.RandomGamma(p=1),    
        ]),
        # a.Compose([
        #     a.CLAHE(),
        #     a.RandomRotate90(),
        #     a.Transpose(),
        #     a.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
        #     a.Blur(blur_limit=3),
        #     a.OpticalDistortion(),
        #     a.GridDistortion(),
        #     a.HueSaturationValue(),
        # ]),
    #     a.Compose([    
    #     a.VerticalFlip(p=0.5),              
    #     a.RandomRotate90(p=0.5),
    #     a.OneOf([
    #         a.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
    #         a.GridDistortion(p=0.5),
    #         a.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)                  
    #         ], p=0.8),
    #     a.CLAHE(p=0.8),
    #     a.RandomBrightnessContrast(p=0.8),    
    #     a.RandomGamma(p=0.8)])
    ]

    return transformations

def _augment_image(img, mask, save = False):
    #img -> image route
    #mask -> mask route. spected a numpy array
    img_t = []
    mask_t = []
    for transform in _get_transformations():
        transformed = transform(image = img, mask = mask)
        img_t.append(transformed['image']) 
        mask_t.append(transformed['mask'])

    return img_t, mask_t

def augment_dataset(dataframe: pd.DataFrame):

    output = pd.DataFrame()
    for row in dataframe.iterrows():
        output = output.append({'img' : row[1]['img'],
                        'mask' : row[1]['mask']}, ignore_index=True)
        aug_img, aug_mask = _augment_image(row[1].img, row[1]['mask'])

        for img, mask in zip(aug_img, aug_mask):
            output = output.append({'img' : img,
                            'mask' : mask}, ignore_index=True)

    return output

def _create_masks(df: pd.DataFrame, img_size):
    output_df = {}
    for row in df.iterrows():
        img = Image.new('L', (img_size, img_size), 0)
        factor = img_size/100
        mask = []
        literal_label = row[1].label
        for label in literal_label:

            if label['polygonlabels'][0] not in LABEL.keys():
                print(f'check annotations on file: {row[1].image}')
                continue

            good_points = [((x1*factor),(x2*factor)) for x1,x2 in label['points']]
            ImageDraw.Draw(img).polygon(good_points,
                                        outline=LABEL[label['polygonlabels'][0]],
                                        fill=LABEL[label['polygonlabels'][0]])
            mask = np.array(img)
        
        output_df[row[1].image] = mask

    return output_df
